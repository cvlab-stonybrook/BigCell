"""Build BigCell training data from large 20x region images.

Input: region images at 20x (e.g. 32768x32768 crops of WSIs, produced with TRIDENT or OpenSlide),
organised as ``<input-root>/<source>/<type>/<slide>_x<X>_y<Y>.{png,jpg,tif}``.

For every region and every size S in {2048, ..., 32768} that fits, non-overlapping SxS crops that
are not mostly background are written to ``<out-root>/<source>/<type>/<slide>_x<X>_y<Y>/<S>/``:

    <N>.jpg        the crop resized to 2048x2048
    <N>_full.jpg   the crop at full resolution (S > 2048 only; skip with --no-full)
    <N>_uni.npy    (S/256, S/256, 1536) UNI-2 grid of the full-resolution crop (fp16)
    <N>_vae.npy    (64, 256, 256) Flux.2 VAE posterior mean || std of <N>.jpg (fp16)

Afterwards build the manifest with ``scripts/build_manifest.py``.

    torchrun --nproc_per_node 4 scripts/extract_features.py --input-root regions/ --out-root bigcell_data/
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
from bigcell.vae import FLUX2_VAE, Flux2VAE  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SIZES = [32768, 16384, 8192, 4096, 2048]
TILE, UNI_PATCH = 2048, 256
VOID_VALUE, HOMOGENEITY_THRESH = 5, 0.7


def load_uni2(device, checkpoint=None):
    """UNI2-h (https://huggingface.co/MahmoodLab/UNI2-h, gated)."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    kwargs = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                  mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                  act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs).eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


@torch.no_grad()
def uni_grid(model, transform, arr, device, batch_size=256):
    """arr: (H, W, 3) uint8 with H, W multiples of 256 -> (H/256, W/256, 1536) fp16."""
    gh, gw = arr.shape[0] // UNI_PATCH, arr.shape[1] // UNI_PATCH
    patches = [Image.fromarray(arr[i * UNI_PATCH:(i + 1) * UNI_PATCH, j * UNI_PATCH:(j + 1) * UNI_PATCH])
               for i in range(gh) for j in range(gw)]
    feats = []
    for s in range(0, len(patches), batch_size):
        x = torch.stack([transform(p) for p in patches[s:s + batch_size]]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(model(x).float().cpu())
    return torch.cat(feats).view(gh, gw, -1).numpy().astype(np.float16)


@torch.no_grad()
def vae_stats(vae, img, device):
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].to(device).float() / 127.5 - 1
    with torch.autocast("cuda", dtype=torch.float16):
        post = vae.encode(x)
    return torch.cat([post.mean, post.std], dim=1)[0].float().cpu().numpy().astype(np.float16)


def valid_bbox(arr):
    mask = arr.max(axis=2) > VOID_VALUE
    if not mask.any():
        return None
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def is_background(crop):
    """Mostly void (padding) or dominated by a single intensity (glass)."""
    if np.all(crop <= VOID_VALUE, axis=-1).mean() > 0.10:
        return True
    flat = (crop // 10).mean(axis=2).astype(np.int32)
    counts = np.unique(flat, return_counts=True)[1]
    return counts.max() / counts.sum() > HOMOGENEITY_THRESH


def process_region(path, rel_dir, args, uni, transform, vae, device):
    arr = np.asarray(Image.open(path).convert("RGB"))
    bbox = valid_bbox(arr)
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    arr = arr[y1:y2, x1:x2]
    h, w = (arr.shape[0] // UNI_PATCH) * UNI_PATCH, (arr.shape[1] // UNI_PATCH) * UNI_PATCH
    if h < TILE or w < TILE:
        return
    arr = arr[:h, :w]
    grid = uni_grid(uni, transform, arr, device)

    for size in SIZES:
        if h < size or w < size:
            continue
        out = args.out_root / rel_dir / str(size)
        out.mkdir(parents=True, exist_ok=True)
        idx = 0
        for y in range(0, h - size + 1, size):
            for x in range(0, w - size + 1, size):
                crop = arr[y:y + size, x:x + size]
                if is_background(crop):
                    continue
                full = Image.fromarray(crop)
                small = full if size == TILE else full.resize((TILE, TILE), resample=Image.BICUBIC)
                g = size // UNI_PATCH
                small.save(out / f"{idx}.jpg", quality=90)
                if size > TILE and not args.no_full:
                    full.save(out / f"{idx}_full.jpg", quality=90)
                np.save(out / f"{idx}_uni.npy", grid[y // UNI_PATCH:y // UNI_PATCH + g, x // UNI_PATCH:x // UNI_PATCH + g])
                np.save(out / f"{idx}_vae.npy", vae_stats(vae, small, device))
                idx += 1


def main(args):
    device, rank, world = setup_distributed()
    regions = sorted(p for ext in ("png", "jpg", "jpeg", "tif", "tiff")
                     for p in args.input_root.glob(f"*/*/*.{ext}"))
    mine = regions[rank::world]
    if rank == 0:
        print(f"{len(regions)} regions, {len(mine)} per rank")
    uni, transform = load_uni2(device, args.uni_checkpoint)
    vae = Flux2VAE(args.vae).to(device)
    for i, path in enumerate(mine):
        rel_dir = path.relative_to(args.input_root).with_suffix("")
        if (args.out_root / rel_dir / str(TILE)).exists() and not args.overwrite:
            continue
        process_region(path, rel_dir, args, uni, transform, vae, device)
        print(f"[rank {rank}] {i + 1}/{len(mine)} {rel_dir}", flush=True)
    if dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--vae", default=FLUX2_VAE)
    p.add_argument("--uni-checkpoint", default=None, help="local UNI2-h pytorch_model.bin (optional)")
    p.add_argument("--no-full", action="store_true", help="do not store full-resolution crops for S > 2048")
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
