"""BigCell tile generator: a SiT flow-matching transformer conditioned on a grid
of UNI-2 embeddings via cross-attention.

References:
    SiT: https://github.com/willisma/SiT
    DiT: https://github.com/facebookresearch/DiT
    REPA-E: https://github.com/End2End-Diffusion/REPA-E
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed
from torch.utils.checkpoint import checkpoint


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, x.ndim)))


#################################################################################
#                                 Building blocks                               #
#################################################################################


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cond):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q_linear(x).reshape(B, N, h, C // h).transpose(1, 2)
        k, v = self.kv_linear(cond).reshape(B, cond.shape[1], 2, h, C // h).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.mlp(self.positional_embedding(t, self.frequency_embedding_size).to(t.dtype))


class CaptionEmbedder(nn.Module):
    """Projects UNI-2 tokens to the hidden size. Holds a learned null embedding for CFG."""

    def __init__(self, in_channels, hidden_size, uncond_prob, token_num):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, drop=0)
        self.register_buffer("y_embedding", torch.randn(token_num, in_channels) / in_channels**0.5)
        self.uncond_prob = uncond_prob

    def forward(self, caption, train):
        if train and self.uncond_prob > 0:
            drop = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
            caption = torch.where(drop[:, None, None], self.y_embedding.to(caption.dtype), caption)
        return self.y_proj(caption)


class SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.cross_attn = CrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, cond, t):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.cross_attn(x, cond)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


#################################################################################
#                                   SiT model                                   #
#################################################################################


class SiT(nn.Module):
    """
    Args:
        input_size: latent spatial size (resolution // 8 for the Flux2 VAE).
        in_channels: latent channels (32 for Flux2).
        num_caption_tokens: number of UNI-2 tokens per tile, (resolution // 256) ** 2.
    """

    def __init__(
        self,
        input_size=256,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        caption_channels=1536,
        num_caption_tokens=64,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.in_channels = self.out_channels = in_channels
        self.num_caption_tokens = num_caption_tokens
        self.gradient_checkpointing = gradient_checkpointing

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = CaptionEmbedder(caption_channels, hidden_size, class_dropout_prob, num_caption_tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)

        # Fixed sin-cos embedding for conditioning tokens, placed at the centre of the latent region each token covers.
        cond_grid = int(num_caption_tokens**0.5)
        if num_caption_tokens > 1 and cond_grid**2 == num_caption_tokens:
            self.y_pos_embed = nn.Parameter(torch.zeros(1, num_caption_tokens, hidden_size), requires_grad=False)
        else:
            self.y_pos_embed = None

        self.blocks = nn.ModuleList([SiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        grid_size = int(self.x_embedder.num_patches**0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.y_pos_embed is not None:
            cond_grid = int(self.num_caption_tokens**0.5)
            # NOTE: kept as in the trained models (uses len(pos_embed[0]) = number of patches)
            scale = self.x_embedder.num_patches // cond_grid
            coords = np.arange(cond_grid, dtype=np.float32) * scale + scale / 2.0 - 0.5
            grid = np.stack(np.meshgrid(coords, coords), axis=0).reshape(2, -1)
            y_pos = get_2d_sincos_pos_embed_from_grid(self.y_pos_embed.shape[-1], grid)
            self.y_pos_embed.data.copy_(torch.from_numpy(y_pos).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c, p = self.out_channels, self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    @property
    def y_null(self):
        """Unconditional embedding used for classifier-free guidance, shape (1, T, C)."""
        return self.y_embedder.y_embedding[None]

    def forward(self, x, t, y):
        """Predict the velocity.

        x: (B, C, H, W) noisy latent, t: (B,) time in [0, 1] (1 = noise), y: (B, T, 1536) UNI-2 tokens.
        """
        x = self.x_embedder(x) + self.pos_embed
        t_embed = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        if self.y_pos_embed is not None:
            y = y + self.y_pos_embed

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, y, t_embed, use_reentrant=False)
            else:
                x = block(x, y, t_embed)

        x = self.final_layer(x, t_embed)
        return self.unpatchify(x)

    @torch.no_grad()
    def forward_with_cfg(self, x, t, y, cfg_scale):
        v_cond = self(x, t, y)
        if cfg_scale == 1.0:
            return v_cond
        v_uncond = self(x, t, self.y_null.expand_as(y).to(y.dtype))
        return v_uncond + cfg_scale * (v_cond - v_uncond)


#################################################################################
#                       Flow-matching training objective                        #
#################################################################################


def sample_time(batch_size, weighting="uniform", alpha_shift=1.0, device="cpu"):
    """Training time distribution. `shifted_uniform` is the SD3-style shift used at >=1024px."""
    u = torch.rand(batch_size, device=device)
    if weighting == "uniform":
        return u
    if weighting == "shifted_uniform":
        return alpha_shift * u / (1 + (alpha_shift - 1) * u)
    raise ValueError(f"unknown weighting {weighting}")


def flow_matching_loss(model, x0, y, weighting="uniform", alpha_shift=1.0):
    """Linear interpolant x_t = (1 - t) x0 + t eps, velocity target eps - x0."""
    t = sample_time(x0.shape[0], weighting, alpha_shift, device=x0.device)
    noise = torch.randn_like(x0)
    t_ = t.view(-1, 1, 1, 1).to(x0.dtype)
    x_t = (1 - t_) * x0 + t_ * noise
    v = model(x_t, t.to(x0.dtype), y)
    return mean_flat((v - (noise - x0)) ** 2)


#################################################################################
#                   Sine/Cosine positional embedding (from MAE)                 #
#################################################################################


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    coords = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(coords, coords), axis=0).reshape(2, 1, grid_size, grid_size)
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


#################################################################################
#                                    Configs                                    #
#################################################################################


def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


SiT_models = {"SiT-XL/2": SiT_XL_2}


def build_sit(resolution, model="SiT-XL/2", **kwargs):
    """Build the tile generator for a given native pixel resolution (Flux2 VAE, UNI-2 tokens)."""
    return SiT_models[model](
        input_size=resolution // 8,
        in_channels=32,
        num_caption_tokens=(resolution // 256) ** 2,
        caption_channels=1536,
        **kwargs,
    )
