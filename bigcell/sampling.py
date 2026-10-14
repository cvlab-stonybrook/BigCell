"""Sampling for BigCell.

* ``sample_tile``: Euler ODE sampler for a single native-resolution tile.
* ``sliding_window_sample``: BigCell-large. Denoises an arbitrarily large latent canvas by
  running the tile generator on overlapping windows and averaging their velocities
  (flow-matching MultiDiffusion, Appendix C). Optionally uses adaptive scheduling
  (Sec. 3.3): after the first step only the top-r fraction of windows, ranked by the
  CFG magnitude ``|v_cond - v_uncond|`` from their last evaluation, are recomputed; the
  rest reuse their cached velocity. Windows are sharded across GPUs when
  ``torch.distributed`` is initialised.
* ``tiled_vae_decode``: decodes a large latent canvas tile by tile.
"""

import math
import random
import time

import torch
import torch.distributed as dist
from einops import rearrange

LATENTS_PER_TOKEN = 32  # one UNI-2 token covers 256 px = 32 Flux2 latents


def time_steps(num_steps, alpha_shift=1.0, device="cpu"):
    """Times from 1 (noise) to 0 (data), with the same shift as used in training."""
    t = torch.linspace(1, 0, num_steps + 1, dtype=torch.float32, device=device)
    if alpha_shift != 1.0:
        t = alpha_shift * t / (1 + (alpha_shift - 1) * t)
    return t


@torch.no_grad()
def sample_tile(model, y, num_steps=20, cfg_scale=2.0, alpha_shift=1.0, generator=None, dtype=torch.bfloat16):
    """y: (B, T, 1536) UNI-2 tokens. Returns latents (B, 32, H, W)."""
    device = y.device
    size = model.x_embedder.img_size[0]
    x = torch.randn(y.shape[0], model.in_channels, size, size, device=device, generator=generator)
    t_steps = time_steps(num_steps, alpha_shift, device)
    with torch.autocast("cuda", dtype=dtype):
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            v = model.forward_with_cfg(x, t_cur.expand(x.shape[0]), y, cfg_scale)
            x = x + (t_next - t_cur) * v.float()
    return x


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def gaussian_kernel(size, sigma=0.5, device="cpu"):
    x = torch.linspace(-1, 1, size)
    x = torch.stack((x.tile(size, 1), x.tile(size, 1).T), dim=0)
    k = torch.exp(-(torch.linalg.norm(x, dim=0) ** 2) / (2 * sigma**2))
    return (k / k.max()).to(device)


def enumerate_windows(H, W, win_size, stride):
    return [(h, w) for h in range(0, H - win_size + 1, stride) for w in range(0, W - win_size + 1, stride)]


def window_condition(uni_grid, h, w, tokens_per_window):
    """uni_grid: (B, Hg, Wg, 1536). Returns the (B, T, 1536) tokens under latent window (h, w)."""
    hi, wi = h // LATENTS_PER_TOKEN, w // LATENTS_PER_TOKEN
    y = uni_grid[:, hi:hi + tokens_per_window, wi:wi + tokens_per_window]
    pad_h, pad_w = tokens_per_window - y.shape[1], tokens_per_window - y.shape[2]
    if pad_h or pad_w:
        y = torch.nn.functional.pad(y, (0, 0, 0, pad_w, 0, pad_h))
    return rearrange(y, "b h w c -> b (h w) c")


def select_windows(windows, importance, top_r, step_idx, strategy="importance", seed=0):
    if step_idx == 1 or top_r >= 1.0:
        return windows
    n = max(1, math.ceil(top_r * len(windows)))
    if strategy == "random":
        return random.Random(seed + step_idx).sample(windows, n)
    return sorted(windows, key=lambda w: -importance.get(w, 0.0))[:n]


@torch.no_grad()
def sliding_window_sample(
    model,
    uni_grid,
    num_steps=20,
    cfg_scale=2.0,
    alpha_shift=1.0,
    stride=128,
    sigma=0.5,
    top_r=1.0,
    window_selection="importance",
    seed=0,
    dtype=torch.bfloat16,
    verbose=True,
):
    """Generate a latent canvas conditioned on a UNI-2 grid of shape (1, Hg, Wg, 1536).

    The canvas is (1, 32, 32*Hg, 32*Wg), i.e. 256 px per grid cell after decoding.
    ``top_r`` is the fraction of windows recomputed at every step after the first.
    All ranks must call this with the same inputs; they return the same canvas.
    """
    device = uni_grid.device
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)

    win = model.x_embedder.img_size[0]
    tokens_per_window = win // LATENTS_PER_TOKEN
    H, W = uni_grid.shape[1] * LATENTS_PER_TOKEN, uni_grid.shape[2] * LATENTS_PER_TOKEN
    windows = enumerate_windows(H, W, win, stride)
    if not windows:
        raise ValueError(f"canvas {H}x{W} latents is smaller than one window ({win})")

    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, model.in_channels, H, W, device=device, generator=g)
    kernel = gaussian_kernel(win, sigma, device)
    y_null = model.y_null.to(device)
    t_steps = time_steps(num_steps, alpha_shift, device)

    importance = {}
    v_prev = torch.zeros_like(x)
    n_evals = 0
    for step_idx, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:]), start=1):
        t0 = time.time()
        active = select_windows(windows, importance, top_r, step_idx, window_selection, seed)
        v_sum, w_sum = torch.zeros_like(x), torch.zeros_like(x)
        local_importance = {}
        t = t_cur.reshape(1)
        with torch.autocast("cuda", dtype=dtype):
            for h, w in active[rank::world]:
                x_win = x[:, :, h:h + win, w:w + win]
                y = window_condition(uni_grid, h, w, tokens_per_window)
                v_cond = model(x_win, t, y).float()
                v_uncond = model(x_win, t, y_null).float()
                local_importance[(h, w)] = (v_cond - v_uncond).abs().mean().item()
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
                v_sum[:, :, h:h + win, w:w + win] += v * kernel
                w_sum[:, :, h:h + win, w:w + win] += kernel
        if world > 1:
            dist.all_reduce(v_sum)
            dist.all_reduce(w_sum)
            gathered = [None] * world
            dist.all_gather_object(gathered, local_importance)
            for d in gathered:
                importance.update(d)
        else:
            importance.update(local_importance)

        # windows that were skipped this step keep their cached velocity
        v = torch.where(w_sum > 0, v_sum / w_sum.clamp(min=1e-6), v_prev)
        v_prev = v
        x = x + (t_next - t_cur) * v
        n_evals += 2 * len(active)
        if verbose and rank == 0:
            print(f"  step {step_idx}/{num_steps}: {len(active)}/{len(windows)} windows, {time.time() - t0:.1f}s")
    return x, n_evals


@torch.no_grad()
def tiled_vae_decode(vae, latents, tile_size=256, sigma=0.5, dtype=torch.bfloat16):
    """Decode a (1, 32, H, W) latent canvas in overlapping tiles. Returns a uint8 (H*8, W*8, 3) tensor
    on rank 0 (None on other ranks). Accumulates on CPU so that 32k images fit in memory."""
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    device = latents.device
    _, _, h, w = latents.shape
    tile_size = min(tile_size, h, w)
    out = torch.zeros(3, h * 8, w * 8)
    count = torch.zeros(1, h * 8, w * 8)
    kernel = gaussian_kernel(tile_size * 8, sigma).view(1, tile_size * 8, tile_size * 8)
    stride = tile_size // 2
    tiles = enumerate_windows(h, w, tile_size, stride)
    for i, j in tiles[rank::world]:
        with torch.autocast("cuda", dtype=dtype):
            dec = vae.decode(latents[:, :, i:i + tile_size, j:j + tile_size])[0].float().cpu()
        sl = (slice(None), slice(i * 8, (i + tile_size) * 8), slice(j * 8, (j + tile_size) * 8))
        out[sl] += dec * kernel
        count[sl] += kernel

    if world > 1:
        chunk = 4096
        for s in range(0, h * 8, chunk):
            o, c = out[:, s:s + chunk].to(device), count[:, s:s + chunk].to(device)
            dist.reduce(o, dst=0)
            dist.reduce(c, dst=0)
            if rank == 0:
                out[:, s:s + chunk], count[:, s:s + chunk] = o.cpu(), c.cpu()
    if rank != 0:
        return None
    img = (out / count.clamp(min=1e-6) + 1) / 2
    return (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
