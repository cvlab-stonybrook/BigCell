"""Generate a large image (BigCell-large) from a UNI-2 conditioning grid.

The grid is either extracted from a real slide region (``<N>_uni.npy`` in a bigcell_data
``<size>/`` folder) or sampled from text with the CGM (``scripts/generate_from_text.py``).
The output is (256 * Hg) x (256 * Wg) pixels; e.g. a 32x32 grid gives an 8192x8192 image.

    # 8k on one GPU
    python scripts/generate_from_grid.py --grid data/.../8192/0_uni.npy --out out/8k.jpg
    # 32k, windows sharded over 4 GPUs, adaptive scheduling with r = 0.25
    torchrun --nproc_per_node 4 scripts/generate_from_grid.py \
        --grid data/.../32768/0_uni.npy --top-r 0.25 --out out/32k.jpg --tiff
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigcell.distributed import cleanup_distributed, setup_distributed  # noqa: E402
from bigcell.models import load_bigcell  # noqa: E402
from bigcell.sampling import sliding_window_sample, tiled_vae_decode  # noqa: E402
from bigcell.vae import FLUX2_VAE, Flux2VAE  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main(args):
    device, rank, _ = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True

    model, config = load_bigcell(args.model, device)
    vae = Flux2VAE(args.vae).to(device)

    grid = torch.from_numpy(np.load(args.grid)).float()  # (Hg, Wg, 1536)
    grid = grid[None].to(device)
    if rank == 0:
        h, w = grid.shape[1] * 256, grid.shape[2] * 256
        print(f"{args.grid}: grid {tuple(grid.shape[1:3])} -> {h}x{w} px, top_r={args.top_r}")

    t0 = time.time()
    latents, n_evals = sliding_window_sample(
        model, grid,
        num_steps=args.num_steps, cfg_scale=args.cfg_scale, alpha_shift=config["alpha_shift"],
        stride=args.stride, sigma=args.sigma, top_r=args.top_r,
        window_selection=args.window_selection, seed=args.seed,
    )
    t_denoise = time.time() - t0
    img = tiled_vae_decode(vae, latents)

    if rank == 0:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        img = img.numpy()
        Image.fromarray(img).save(out, quality=90)
        if args.tiff:
            import tifffile

            tifffile.imwrite(out.with_suffix(".tiff"), img, tile=(512, 512), compression="jpeg", bigtiff=True)
        info = dict(grid=str(args.grid), top_r=args.top_r, model_evals=n_evals,
                    denoise_s=round(t_denoise, 1), total_s=round(time.time() - t0, 1))
        print(json.dumps(info))
    cleanup_distributed()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True, help="(Hg, Wg, 1536) UNI-2 grid, e.g. <region>/8192/0_uni.npy")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="bigcell-2048", help="released model name or local directory")
    p.add_argument("--vae", default=FLUX2_VAE, help="Flux.2 VAE: HF repo id or local diffusers directory")
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--stride", type=int, default=128, help="window stride in latents (128 = 1024 px)")
    p.add_argument("--sigma", type=float, default=0.5, help="width of the Gaussian window weighting")
    p.add_argument("--top-r", type=float, default=1.0, help="fraction of windows recomputed per step (1 = all)")
    p.add_argument("--window-selection", default="importance", choices=["importance", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tiff", action="store_true", help="also write a tiled BigTIFF")
    main(p.parse_args())
