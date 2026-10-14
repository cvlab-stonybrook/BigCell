"""Convert a BigCell training checkpoint into the released format (EMA weights only): the ``transformer/``
folder of a BigCellPipeline (config.json + diffusion_pytorch_model.safetensors). The output directory can be
passed to ``--model`` / ``--init-from`` and to ``bigcell.models.load_bigcell``.

    python scripts/convert_sit_checkpoint.py --ckpt exps/bigcell_2048/checkpoints/0046800.pt \
        --resolution 2048 --alpha-shift 3.0 --out weights/bigcell-2048
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigcell.models.sit import build_sit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--alpha-shift", type=float, default=1.0, help="time shift used in training (sampler must match)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["ema"]
    model = build_sit(args.resolution)
    model.load_state_dict(state, strict=True)  # fail loudly on any mismatch

    tdir = args.out / "transformer"
    tdir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, tdir / "diffusion_pytorch_model.safetensors")
    config = {"_class_name": "BigCellTransformer2DModel", "sample_size": args.resolution // 8, "patch_size": 2,
              "in_channels": 32, "hidden_size": 1152, "num_layers": 28, "num_attention_heads": 16, "mlp_ratio": 4.0,
              "caption_channels": 1536, "num_caption_tokens": (args.resolution // 256) ** 2,
              "alpha_shift": args.alpha_shift}
    (tdir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved {args.out}  ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")


if __name__ == "__main__":
    main()
