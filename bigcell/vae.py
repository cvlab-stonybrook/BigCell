"""Flux.2 VAE (8x spatial compression, 32 latent channels).

BigCell operates on the *raw* (un-normalised, un-patchified) Flux.2 VAE latent:
a 2048x2048 RGB tile in [-1, 1] maps to a 32x256x256 latent.

The weights are not redistributed with BigCell; by default they are pulled from the
Hugging Face hub (``black-forest-labs/FLUX.2-dev``, subfolder ``vae``; gated, accept the
licence on the model page first). A local diffusers directory can be passed instead.
"""

import torch
import torch.nn as nn

FLUX2_VAE = "black-forest-labs/FLUX.2-dev"


class Flux2VAE(nn.Module):
    def __init__(self, path=FLUX2_VAE):
        super().__init__()
        from diffusers import AutoencoderKLFlux2

        kwargs = {} if _is_local(path) else {"subfolder": "vae"}
        self.vae = AutoencoderKLFlux2.from_pretrained(path, **kwargs)
        self.vae.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, x):
        """x: (B, 3, H, W) in [-1, 1]. Returns the posterior (``.mean``, ``.std``, ``.sample()``)."""
        return self.vae.encode(x).latent_dist

    @torch.no_grad()
    def decode(self, z):
        """z: (B, 32, H/8, W/8). Returns images in [-1, 1]."""
        return self.vae.decode(z).sample


def _is_local(path):
    from pathlib import Path

    return Path(path).exists()
