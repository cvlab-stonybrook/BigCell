import json
from pathlib import Path

from .models import CGM_B, GridNormalizer

HF_REPO = "StonyBrook-CVLab/BigCell"


def load_cgm(path="cgm-8192", device="cpu"):
    """Load the released CGM: a local directory or the ``cgm-8192`` subfolder of the BigCell HF repo.
    Returns ``(model, normalizer, config)``."""
    from safetensors.torch import load_file

    local = Path(path)
    if not local.exists():
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(HF_REPO, allow_patterns=[f"{path}/*"])) / path
    config = json.loads((local / "config.json").read_text())
    model = CGM_B(input_size=config["grid_size"])
    model.load_state_dict(load_file(local / "model.safetensors", device="cpu"))
    normalizer = GridNormalizer(local / "normalization_stats.safetensors")
    return model.to(device).eval(), normalizer.to(device), config
