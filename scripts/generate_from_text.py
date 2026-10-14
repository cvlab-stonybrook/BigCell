"""Text -> 8192x8192 image: TITAN text encoder -> CGM (UNI-2 grid) -> BigCell-large.

    python scripts/generate_from_text.py --out out/pdac \
        --prompt "Pancreatic ductal adenocarcinoma, moderately differentiated, with perineural invasion."
    # several samples per prompt, windows sharded across 4 GPUs
    torchrun --nproc_per_node 4 scripts/generate_from_text.py --prompt "..." --num-samples 2 --out out/x

Writes <out>/sample_<i>.jpg and the generated grid <out>/sample_<i>_uni.npy (which can be
re-rendered with scripts/generate_from_grid.py).
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

Image.MAX_IMAGE_PIXELS = None


@torch.no_grad()
def sample_grid(cgm, normalizer, config, text_tokens, num_steps, cfg_scale, seed, device):
    """text_tokens: (T, 768). Returns a raw (Hg, Wg, 1536) UNI-2 grid."""
    g = config["grid_size"]
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 1536, g, g, device=device, generator=gen)
    y = torch.as_tensor(text_tokens, device=device).float()[None]
    times = sampling_times(num_steps, config["time_shift"], device=device)
    z = euler_sample(cgm, x, y, times, cfg_scale)
    return normalizer.decode(z)[0].permute(1, 2, 0)


def main(args):
    device, rank, world = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    # 1. text -> TITAN tokens (rank 0, then broadcast)
    if rank == 0:
        titan = load_titan(args.titan, device)
        tokens = torch.from_numpy(encode_text(titan, [args.prompt], device)[1][0]).to(device)
        del titan
        torch.cuda.empty_cache()
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

    for i in range(args.num_samples):
        seed = args.seed + i
        # 2. text -> UNI-2 grid (identical on all ranks: same seed)
        grid = sample_grid(cgm, normalizer, cgm_config, tokens, args.cgm_steps, args.cgm_cfg_scale, seed, device)
        # 3. grid -> image
        latents, _ = sliding_window_sample(
            model, grid[None], num_steps=args.num_steps, cfg_scale=args.cfg_scale,
            alpha_shift=config["alpha_shift"], top_r=args.top_r, seed=seed, verbose=False,
        )
        img = tiled_vae_decode(vae, latents)
        if rank == 0:
            np.save(out / f"sample_{i}_uni.npy", grid.half().cpu().numpy())
            Image.fromarray(img.numpy()).save(out / f"sample_{i}.jpg", quality=90)
            print(f"saved {out / f'sample_{i}.jpg'}")
    if rank == 0:
        (out / "prompt.txt").write_text(args.prompt + "\n")
    cleanup_distributed()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cgm", default="cgm-8192", help="released CGM name or local directory")
    p.add_argument("--cgm-steps", type=int, default=50)
    p.add_argument("--cgm-cfg-scale", type=float, default=2.0)
    p.add_argument("--titan", default=TITAN, help="TITAN: HF repo id or local directory")
    p.add_argument("--model", default="bigcell-2048", help="released tile generator or local directory")
    p.add_argument("--vae", default=FLUX2_VAE)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--top-r", type=float, default=1.0)
    main(p.parse_args())
