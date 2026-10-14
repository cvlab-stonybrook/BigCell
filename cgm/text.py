"""TITAN text encoder (https://huggingface.co/MahmoodLab/TITAN, gated; request access first).

The CGM is conditioned on the *unpooled* token embeddings of the text encoder, (T, 768).
"""

import numpy as np
import torch

TITAN = "MahmoodLab/TITAN"


def load_titan(path=TITAN, device="cuda"):
    from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizerFast

    # TITAN's remote code was written for older transformers releases; these shims keep it loading.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    if not hasattr(PreTrainedTokenizerFast, "batch_encode_plus"):
        PreTrainedTokenizerFast.batch_encode_plus = lambda self, texts, **kw: self(texts, **kw)
    orig_linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):  # remote code calls linspace on the meta device during init
        kwargs.pop("device", None)
        return orig_linspace(*args, **kwargs, device="cpu")

    torch.linspace = cpu_linspace
    try:
        titan = AutoModel.from_pretrained(path, trust_remote_code=True)
    finally:
        torch.linspace = orig_linspace
    return titan.to(device).eval()


@torch.no_grad()
def encode_text(titan, texts, device="cuda"):
    """Returns (pooled (N, 768), tokens (N, T, 768)) as float32 numpy arrays."""
    ids = titan.text_encoder.tokenizer(list(texts)).to(device)[:, :-1]
    pooled, tokens = titan.text_encoder(ids)
    return pooled.float().cpu().numpy(), tokens.float().cpu().numpy()


def encode_and_save(titan, texts, out_path, device="cuda"):
    """Encode several texts describing the same case (e.g. report paraphrases) into
    ``<out_path>`` as an (N, T, 768) float16 array, the format read by the CGM dataset."""
    _, tokens = encode_text(titan, texts, device)
    np.save(out_path, tokens.astype(np.float16))
    return tokens
