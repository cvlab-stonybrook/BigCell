"""Conditioning generation model (CGM): a DiT-DH (DiT with a wide DDT head) flow-matching
transformer that generates a grid of UNI-2 embeddings conditioned on TITAN text tokens
through cross-attention.

Adapted from RAE (https://github.com/bytetriper/RAE, MIT licence), which builds on
LightningDiT and DDT. Rotary embeddings follow EVA (https://github.com/baaivision/EVA).
"""

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed


#################################################################################
#                                 Building blocks                               #
#################################################################################


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    """2D axial RoPE over a pt_seq_len x pt_seq_len grid (EVA "fast" variant)."""

    def __init__(self, dim, pt_seq_len, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        t = torch.arange(pt_seq_len).float()
        freqs = repeat(torch.einsum("i,f->if", t, freqs), "... n -> ... (n r)", r=2)
        freqs = torch.cat(
            [freqs[:, None, :].expand(-1, pt_seq_len, -1), freqs[None, :, :].expand(pt_seq_len, -1, -1)], dim=-1
        )
        self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]))
        self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]))

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        if rope is not None:
            q, k = rope(q).to(v.dtype), rope(k).to(v.dtype)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        return self.mlp(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))


class TextEmbedder(nn.Module):
    """Projects (B, T, 768) TITAN text tokens to the hidden size. During training the whole
    sequence is replaced by learned null tokens with probability ``dropout_prob`` (for CFG)."""

    def __init__(self, titan_dim, hidden_size, dropout_prob):
        super().__init__()
        self.proj = nn.Linear(titan_dim, hidden_size, bias=True)
        self.null_tokens = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout_prob = dropout_prob

    def forward(self, y, train, force_drop_ids=None):
        emb = self.proj(y)
        if force_drop_ids is not None:
            drop = force_drop_ids == 1
        elif train and self.dropout_prob > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
        else:
            return emb
        return torch.where(drop[:, None, None], self.null_tokens.expand_as(emb).to(emb.dtype), emb)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DDTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cross_attn=False):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.has_cross_attn = cross_attn
        if cross_attn:
            self.norm_cross = RMSNorm(hidden_size)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            nn.init.constant_(self.cross_attn.out_proj.weight, 0)
            nn.init.constant_(self.cross_attn.out_proj.bias, 0)
        self.attn = Attention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, rope=None, text_kv=None):
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope)
        if self.has_cross_attn and text_kv is not None:
            x = x + self.cross_attn(self.norm_cross(x), text_kv, text_kv, need_weights=False)[0]
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


#################################################################################
#                                     CGM                                       #
#################################################################################


class CGM(nn.Module):
    """DiT-DH operating on a (C=1536, H, W) grid of normalised UNI-2 embeddings (patch size 1).

    The encoder (``depth[0]`` blocks, width ``hidden_size[0]``) sees the text through cross-attention;
    the DDT head (``depth[1]`` blocks, width ``hidden_size[1]``) is modulated per token by the encoder output.
    """

    def __init__(
        self,
        input_size=32,
        in_channels=1536,
        hidden_size=(768, 2048),
        depth=(12, 2),
        num_heads=(12, 16),
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        titan_dim=768,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        enc_dim, dec_dim = hidden_size
        self.num_encoder_blocks, self.num_blocks = depth[0], depth[0] + depth[1]

        self.x_embedder = PatchEmbed(input_size, 1, in_channels, dec_dim, bias=True)
        self.s_embedder = PatchEmbed(input_size, 1, in_channels, enc_dim, bias=True)
        self.s_projector = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        self.t_embedder = GaussianFourierEmbedding(enc_dim)
        self.y_embedder = TextEmbedder(titan_dim, enc_dim, class_dropout_prob)
        self.final_layer = FinalLayer(dec_dim, in_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size**2, enc_dim), requires_grad=False)
        self.enc_feat_rope = VisionRotaryEmbedding(enc_dim // num_heads[0] // 2, input_size)
        self.dec_feat_rope = VisionRotaryEmbedding(dec_dim // num_heads[1] // 2, input_size)
        self.blocks = nn.ModuleList(
            [DDTBlock(enc_dim, num_heads[0], mlp_ratio, cross_attn=True) for _ in range(depth[0])]
            + [DDTBlock(dec_dim, num_heads[1], mlp_ratio) for _ in range(depth[1])]
        )
        self.initialize_weights()

    def initialize_weights(self):
        for emb in (self.x_embedder, self.s_embedder):
            w = emb.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(emb.proj.bias, 0)
        nn.init.normal_(self.y_embedder.proj.weight, std=0.02)
        self.pos_embed.data.copy_(
            torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.input_size)).float().unsqueeze(0)
        )
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, force_drop_ids=None):
        """x: (B, 1536, H, W) noisy normalised grid, t: (B,) time in [0, 1] (1 = noise),
        y: (B, T, 768) TITAN text tokens. Returns the velocity (B, 1536, H, W)."""
        H = x.shape[-1]
        t = self.t_embedder(t)
        text_kv = self.y_embedder(y, self.training, force_drop_ids)

        s = self.s_embedder(x) + self.pos_embed
        for block in self.blocks[: self.num_encoder_blocks]:
            s = block(s, t, rope=self.enc_feat_rope, text_kv=text_kv)
        s = F.silu(t.unsqueeze(1) + s)
        s = self.s_projector(s)

        x = self.x_embedder(x)
        for block in self.blocks[self.num_encoder_blocks:]:
            x = block(x, s, rope=self.dec_feat_rope)
        x = self.final_layer(x, s)
        return rearrange(x, "b (h w) c -> b c h w", h=H)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        if cfg_scale == 1.0:
            return self(x, t, y)
        n = x.shape[0]
        drop = torch.cat([torch.zeros(n), torch.ones(n)]).long().to(x.device)
        v = self(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, y]), force_drop_ids=drop)
        v_cond, v_uncond = v.chunk(2)
        return v_uncond + cfg_scale * (v_cond - v_uncond)


def CGM_B(input_size=32, **kwargs):
    """DiT-DH-B used in the paper (~360M parameters)."""
    return CGM(input_size=input_size, hidden_size=(768, 2048), depth=(12, 2), num_heads=(12, 16), **kwargs)


#################################################################################
#                     Grid normalisation (per position and channel)             #
#################################################################################


class GridNormalizer(nn.Module):
    """Standardises a UNI-2 grid with per-position, per-channel statistics computed on the
    training set (``image_embedding_stats/uni2_<size>/normalization_stats.pt``, or ``.safetensors``)."""

    def __init__(self, stats_path, eps=1e-5):
        super().__init__()
        if str(stats_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            stats = load_file(stats_path)
        else:
            stats = torch.load(stats_path, map_location="cpu")
        self.register_buffer("mean", stats["mean"].float())  # (1536, H, W)
        self.register_buffer("var", stats["var"].float())
        self.eps = eps

    def _stats(self, z):
        mean, var = self.mean, self.var
        if mean.shape[-2:] != z.shape[-2:]:
            mean = F.interpolate(mean[None], size=z.shape[-2:], mode="bilinear", align_corners=False)[0]
            var = F.interpolate(var[None], size=z.shape[-2:], mode="bilinear", align_corners=False)[0]
        return mean, var

    def encode(self, z):
        """z: (B, 1536, H, W) raw UNI-2 grid -> normalised."""
        mean, var = self._stats(z)
        return (z - mean) / torch.sqrt(var + self.eps)

    def decode(self, z):
        mean, var = self._stats(z)
        return z * torch.sqrt(var + self.eps) + mean


#################################################################################
#                       Flow matching (linear interpolant)                      #
#################################################################################


def time_shift(t, shift):
    return shift * t / (1 + (shift - 1) * t)


def default_time_shift(grid_size, channels=1536, base=4096):
    """Dimension-dependent shift sqrt(dim / 4096) from the paper (Appendix D.2)."""
    return sqrt(channels * grid_size * grid_size / base)


def flow_matching_loss(model, x0, y, shift):
    """x0: normalised grid. Logit-normal(0, 1) time sampling followed by the dimension shift."""
    t = time_shift(torch.sigmoid(torch.randn(x0.shape[0], device=x0.device)), shift)
    noise = torch.randn_like(x0)
    t_ = t.view(-1, 1, 1, 1)
    x_t = (1 - t_) * x0 + t_ * noise
    v = model(x_t, t, y)
    return ((v.float() - (noise - x0)) ** 2).mean(dim=(1, 2, 3))


def sampling_times(num_points, shift, t_start=1.0, device="cpu"):
    """Shifted time grid from t_start to 0. ``num_points`` time points over [1, 0] (i.e. num_points - 1
    Euler steps for a full trajectory), as in the released model's evaluation. With ``t_start < 1``
    only the part of the grid below t_start is kept (used for augmentation, Sec. 3.5)."""
    t = time_shift(torch.linspace(1, 0, num_points), shift)
    if t_start < 1.0:
        t = torch.cat([torch.tensor([t_start]), t[t < t_start]])
    return t.to(device)


@torch.no_grad()
def euler_sample(model, x, y, times, cfg_scale=2.0):
    for t_cur, t_next in zip(times[:-1], times[1:]):
        v = model.forward_with_cfg(x, t_cur.expand(x.shape[0]), y, cfg_scale)
        x = x + (t_next - t_cur) * v.float()
    return x


#################################################################################
#                          Sine/cosine position embedding                       #
#################################################################################


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    coords = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(coords, coords), axis=0).reshape(2, -1)
    return np.concatenate([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], axis=1)


def _sincos_1d(embed_dim, pos):
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
