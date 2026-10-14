import json
from pathlib import Path

from .sit import SiT, build_sit

HF_REPO = "StonyBrook-CVLab/BigCell"


def resolve_model_dir(path):
    """A local directory, or the name of a subfolder of the BigCell HF repo (downloaded on first use)."""
    local = Path(path)
    if local.exists():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_REPO, allow_patterns=[f"{path}/transformer/*"])) / path


def transformer_weights(path):
    """Path of the transformer weights inside a released pipeline folder (or a direct .safetensors file)."""
    path = Path(path)
    if path.suffix == ".safetensors":
        return path
    return path / "transformer" / "diffusion_pytorch_model.safetensors"


def load_bigcell(path="bigcell-2048", device="cpu"):
    """Load a released tile generator (``bigcell-256``, ``bigcell-1024``, ``bigcell-2048`` or a local copy of
    one of these diffusers folders). Returns ``(model, config)``; ``config["alpha_shift"]`` is the time shift
    the model was trained with and must be passed to the samplers."""
    from safetensors.torch import load_file

    root = resolve_model_dir(path)
    cfg = json.loads((root / "transformer" / "config.json").read_text())
    resolution = cfg["sample_size"] * 8
    model = build_sit(resolution)
    model.load_state_dict(load_file(transformer_weights(root), device="cpu"))
    config = {"resolution": resolution, "alpha_shift": cfg["alpha_shift"]}
    return model.to(device).eval(), config
