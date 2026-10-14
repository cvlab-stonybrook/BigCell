"""Class-conditional data augmentation with the CGM (paper Sec. 3.5).

A real region's UNI-2 grid is noised to an intermediate time t' and denoised back with the CGM
conditioned on a class-specific text prompt; the resulting grid is rendered with BigCell-large.
This keeps the global layout of the real region while injecting class-consistent variation.

    torchrun --nproc_per_node 4 scripts/augment.py \
        --grid /path/to/region/8192/0_uni.npy \
        --prompt "Clear cell renal cell carcinoma, Fuhrman grade 2." \
        --t-start 0.5 --num-samples 4 --out out/aug_ccrcc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigcell.distributed import cleanup_distributed, setup_distributed  # noqa: E402
from bigcell.models import load_bigcell  # noqa: E402
from bigcell.sampling import sliding_window_sample, tiled_vae_decode  # noqa: E402
from bigcell.vae import FLUX2_VAE, Flux2VAE  # noqa: E402
from cgm import load_cgm  # noqa: E402
from cgm.models import euler_sample, sampling_times  # noqa: E402
from cgm.text import TITAN, encode_text, load_titan  # noqa: E402


@torch.no_grad()
def morph_grid(cgm, normalizer, config, grid, text_tokens, t_start, num_steps, cfg_scale, seed, device):
    """grid: raw (Hg, Wg, 1536). Returns the augmented raw grid."""
    z0 = normalizer.encode(grid.permute(2, 0, 1)[None])
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z0.shape, device=device, generator=gen)
    z_t = (1 - t_start) * z0 + t_start * noise
    times = sampling_times(num_steps, config["time_shift"], t_start=t_start, device=device)
    y = torch.as_tensor(text_tokens, device=device).float()[None]
    z = euler_sample(cgm, z_t, y, times, cfg_scale)
    return normalizer.decode(z)[0].permute(1, 2, 0)


def main(args):
    device, rank, world = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    if args.prompt_embedding:
        tokens = torch.from_numpy(np.load(args.prompt_embedding).astype(np.float32)).to(device)
        tokens = tokens.reshape(-1, *tokens.shape[-2:])[0]
    else:
        if rank == 0:
            titan = load_titan(args.titan, device)
            tokens = torch.from_numpy(encode_text(titan, [args.prompt], device)[1][0]).to(device)
            del titan
            shape = torch.tensor(tokens.shape, device=device)
        else:
            shape = torch.zeros(2, dtype=torch.long, device=device)
        if world > 1:
            dist.broadcast(shape, 0)
            if rank != 0:
                tokens = torch.empty(*shape.tolist(), device=device)
            dist.broadcast(tokens, 0)

    cgm, normalizer, cgm_config = load_cgm(args.cgm, device)
    model, config = load_bigcell(args.model, device)
    vae = Flux2VAE(args.vae).to(device)
    grid = torch.from_numpy(np.load(args.grid).astype(np.float32)).to(device)
    g = cgm_config["grid_size"]
    if grid.shape[:2] != (g, g):
        raise ValueError(f"grid {tuple(grid.shape[:2])} does not match the CGM grid size {g}x{g}")

    for i in range(args.num_samples):
        seed = args.seed + i
        new_grid = morph_grid(cgm, normalizer, cgm_config, grid, tokens, args.t_start, args.cgm_steps,
                              args.cgm_cfg_scale, seed, device)
        latents, _ = sliding_window_sample(model, new_grid[None], num_steps=args.num_steps,
                                           cfg_scale=args.cfg_scale, alpha_shift=config["alpha_shift"],
                                           top_r=args.top_r, seed=seed, verbose=False)
        img = tiled_vae_decode(vae, latents)
        if rank == 0:
            np.save(out / f"aug_{i}_uni.npy", new_grid.half().cpu().numpy())
            Image.fromarray(img.numpy()).save(out / f"aug_{i}.jpg", quality=90)
            print(f"saved {out / f'aug_{i}.jpg'}")
    cleanup_distributed()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True, help="real (32, 32, 1536) UNI-2 grid of an 8192 px region")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-embedding", default=None, help="pre-computed TITAN tokens (.npy) instead of --prompt")
    p.add_argument("--out", required=True)
    p.add_argument("--t-start", type=float, default=0.5, help="noise level t' (0 = keep the real grid, 1 = from scratch)")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cgm", default="cgm-8192")
    p.add_argument("--cgm-steps", type=int, default=50)
    p.add_argument("--cgm-cfg-scale", type=float, default=2.0)
    p.add_argument("--titan", default=TITAN)
    p.add_argument("--model", default="bigcell-2048")
    p.add_argument("--vae", default=FLUX2_VAE)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--top-r", type=float, default=1.0)
    args = p.parse_args()
    if (args.prompt is None) == (args.prompt_embedding is None):
        p.error("give exactly one of --prompt / --prompt-embedding")
    main(args)
