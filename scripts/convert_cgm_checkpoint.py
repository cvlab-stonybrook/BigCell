"""Convert a CGM training checkpoint into the released format (EMA weights, safetensors, config.json and
the grid normalisation statistics). The output directory can be passed to ``--cgm``.

    python scripts/convert_cgm_checkpoint.py --ckpt exps/cgm_8192/checkpoints/ep-0000080.pt \
        --stats /path/to/bigcell_data/image_embedding_stats/uni2_8192/normalization_stats.pt \
        --grid-size 32 --out weights/cgm-8192
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgm.models import CGM_B, default_time_shift  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ckpt["ema"].items()}
    bad = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise ValueError(f"non-finite values in {len(bad)} tensors, e.g. {bad[:3]}")

    model = CGM_B(input_size=args.grid_size)
    model.load_state_dict(state, strict=True)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, args.out / "model.safetensors")
    stats = torch.load(args.stats, map_location="cpu")
    save_file({k: stats[k].float().contiguous() for k in ("mean", "var")}, args.out / "normalization_stats.safetensors")
    config = {"model": "CGM-B", "grid_size": args.grid_size, "region_size": args.grid_size * 256,
              "time_shift": default_time_shift(args.grid_size), "text_encoder": "MahmoodLab/TITAN",
              "num_inference_steps": 50, "guidance_scale": 2.0}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved {args.out}  ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")


if __name__ == "__main__":
    main()
