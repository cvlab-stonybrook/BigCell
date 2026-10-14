"""Paired (UNI-2 grid, TITAN text) data for training the CGM.

    <root>/metadata/image_list_text_titan.h5      datasets {train,test,missing}_<size>
    <root>/<source>/<type>/<region>/<size>/<N>_uni.npy          (size/256, size/256, 1536)
    <titan_text_root>/<source>/<type>/<case>_unpooled.npy        (K, T, 768): K paraphrased report summaries
                                                                 <case> = TCGA patient barcode or HISTAI case
    <titan_text_root>/null_embed_unpooled.npy                    TITAN tokens of the empty string

Regions listed under ``missing_<size>`` have no report and are paired with the empty-string
embedding (they keep the model's unconditional distribution close to the full data).
"""

import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def case_id(rel_path):
    """Report key of a region: the TCGA patient barcode or the HISTAI case."""
    if rel_path.startswith("histai/"):
        return re.findall(r"(.*)_slide", rel_path)[0]
    return re.findall(r"(.*TCGA-\w\w-\w\w\w\w).*", rel_path)[0]


class GridTextDataset(Dataset):
    def __init__(self, root, titan_text_root, split="train", size=8192, include_missing=True,
                 manifest="image_list_text_titan.h5"):
        self.root = Path(root)
        self.titan_text_root = Path(titan_text_root)
        self.null_embed = np.load(self.titan_text_root / "null_embed_unpooled.npy").astype(np.float32).squeeze(0)
        with h5py.File(self.root / "metadata" / manifest, "r") as f:
            names = [f[f"{split}_{size}"][:]]
            if include_missing:
                names.append(f[f"missing_{size}"][:])
        self.names = np.concatenate(names)
        self.has_text = np.zeros(len(self.names), dtype=bool)
        self.has_text[: len(names[0])] = True

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        rel = self.names[idx].decode()
        img = self.root / rel
        grid = np.load(img.with_name(img.stem + "_uni.npy")).astype(np.float32)  # (H, W, 1536)
        grid = torch.from_numpy(grid).permute(2, 0, 1)
        if self.has_text[idx]:
            texts = np.load(self.titan_text_root / f"{case_id(rel)}_unpooled.npy")
            text = texts[np.random.randint(len(texts))].astype(np.float32)  # random paraphrase
        else:
            text = self.null_embed
        return grid, torch.from_numpy(text)
