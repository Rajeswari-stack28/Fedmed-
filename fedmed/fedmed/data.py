"""Data: BraTS 2021 (real) or a synthetic tumour generator (smoke tests, no download needed).

Channel order of the label follows MONAI's BraTS convention: 0=TC, 1=WT, 2=ET.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import SPATIAL

MODS = ["flair", "t1", "t1ce", "t2"]


class SyntheticBraTS(Dataset):
    """Nested ellipsoid 'tumours' on a smooth noisy brain-ish background."""

    def __init__(self, ids):
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.ids[i])
        zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s) for s in SPATIAL], indexing="ij")
        c = rng.uniform(-0.4, 0.4, 3)
        r = rng.uniform(0.25, 0.5, 3)
        d = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2
        wt, tc, et = d < 1.0, d < 0.5, d < 0.18
        brain = (zz**2 + yy**2 + xx**2) < 0.9
        img = np.stack([
            rng.normal(0, 0.15, SPATIAL) + brain * 1.0 + wt * a + tc * b + et * e
            for a, b, e in [(0.8, 0.4, 0.2), (-0.3, -0.3, -0.2), (0.3, 0.6, 1.0), (0.9, 0.3, 0.1)]
        ]).astype(np.float32)
        img = (img - img.mean((1, 2, 3), keepdims=True)) / (img.std((1, 2, 3), keepdims=True) + 1e-6)
        lab = np.stack([tc, wt, et]).astype(np.float32)
        return {"image": torch.from_numpy(img), "label": torch.from_numpy(lab)}


def list_subjects(root):
    subs = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        item = {m: str(d / f"{d.name}_{m}.nii.gz") for m in MODS}
        item["label"] = str(d / f"{d.name}_seg.nii.gz")
        if all(Path(p).exists() for p in item.values()):
            subs.append(item)
    return subs


def _transforms():
    from monai.transforms import (
        Compose, ConcatItemsd, ConvertToMultiChannelBasedOnBratsClassesd, DeleteItemsd,
        EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd, Orientationd, Resized,
    )
    allk = MODS + ["label"]
    return Compose([
        LoadImaged(keys=allk), EnsureChannelFirstd(keys=allk),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=allk, axcodes="RAS"),
        Resized(keys=MODS, spatial_size=SPATIAL, mode="trilinear"),
        Resized(keys="label", spatial_size=SPATIAL, mode="nearest"),
        ConcatItemsd(keys=MODS, name="image"), DeleteItemsd(keys=MODS),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def get_partition(root, synthetic, num_nodes, seed=42, val_frac=0.2, limit=None, n_synth=16):
    """Deterministic split: shard i belongs to hospital i; each shard keeps its own local val set."""
    if synthetic:
        items = list(range(n_synth * num_nodes))
    else:
        items = list_subjects(root)
        if not items:
            raise SystemExit(f"No BraTS subjects found under {root}. Use --synthetic or fix --data-root.")
    random.Random(seed).shuffle(items)
    if limit:
        items = items[:limit]
    out = []
    for shard in (items[i::num_nodes] for i in range(num_nodes)):
        k = max(1, int(len(shard) * val_frac))
        out.append((shard[k:], shard[:k]))  # (train, val)
    return out


def build_dataset(items, synthetic):
    if synthetic:
        return SyntheticBraTS(items)
    from monai.data import CacheDataset
    return CacheDataset(items, transform=_transforms(), cache_rate=1.0, num_workers=2)
