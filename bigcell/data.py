"""BigCell data layout (see README, "Data"):

    <root>/<source>/<type>/<slide>_x<X>_y<Y>/<size>/<N>.jpg          2048x2048 image (downsampled if size > 2048)
                                                   /<N>_full.jpg     full-resolution image (size > 2048 only)
                                                   /<N>_uni.npy      (size/256, size/256, 1536) UNI-2 grid, fp16
                                                   /<N>_vae.npy      (64, 256, 256) Flux.2 latent mean || std of <N>.jpg
    <root>/metadata/image_list_<split>.h5            datasets ``size_<size>``: relative paths of <N>.jpg

All tiles are at 20x; <size> is the side of the region in pixels at 20x.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

TILE = 2048  # the image model is always trained on 2048 px tiles (cropped for lower resolutions)


class BigCellTileDataset(Dataset):
    """2048 px tiles for training the tile generator.

    For ``resolution < 2048`` each tile is split into ``(2048 / resolution) ** 2`` non-overlapping
    crops and every crop is a separate sample, so one epoch covers each crop once (paper Table 6).

    Returns ``(latent, uni)``: latent (32, res/8, res/8) sampled from the stored VAE posterior,
    uni ((res/256) ** 2, 1536) UNI-2 tokens of the same region.
    """

    def __init__(self, root, split="train", resolution=2048):
        assert TILE % resolution == 0 and resolution % 256 == 0
        self.root = Path(root)
        self.manifest = self.root / "metadata" / f"image_list_{split}.h5"
        self.key = f"size_{TILE}"
        self.resolution = resolution
        self.crops_per_side = TILE // resolution
        with h5py.File(self.manifest, "r") as f:
            self.num_tiles = len(f[self.key])
        self._paths = None  # opened lazily in each worker

    def __len__(self):
        return self.num_tiles * self.crops_per_side**2

    def _path(self, idx):
        if self._paths is None:
            self._paths = h5py.File(self.manifest, "r")[self.key]
        return self.root / self._paths[idx].decode()

    def __getitem__(self, idx):
        tile_idx, crop_idx = divmod(idx, self.crops_per_side**2)
        img_path = self._path(tile_idx)
        uni = np.load(img_path.with_name(img_path.stem + "_uni.npy"))  # (8, 8, 1536)
        vae = np.load(img_path.with_name(img_path.stem + "_vae.npy"))  # (64, 256, 256)

        ci, cj = divmod(crop_idx, self.crops_per_side)
        tok = self.resolution // 256
        lat = self.resolution // 8
        uni = uni[ci * tok:(ci + 1) * tok, cj * tok:(cj + 1) * tok]
        vae = vae[:, ci * lat:(ci + 1) * lat, cj * lat:(cj + 1) * lat]

        mean, std = torch.from_numpy(vae.astype(np.float32)).chunk(2)
        latent = mean + std * torch.randn_like(mean)
        uni = torch.from_numpy(uni.astype(np.float32)).reshape(-1, uni.shape[-1])
        return latent, uni
