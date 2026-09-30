"""3D U-Net (MONAI) for BraTS: 4 MRI modalities in, 3 nested tumour regions out (TC, WT, ET)."""
from collections import OrderedDict

import numpy as np
import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet


def build_model() -> torch.nn.Module:
    # Small on purpose: every parameter is CKKS-encrypted each round (~4096 values per ciphertext).
    return UNet(
        spatial_dims=3, in_channels=4, out_channels=3,
        channels=(8, 16, 32, 64), strides=(2, 2, 2),
        num_res_units=1, norm=Norm.INSTANCE,  # instance norm => no running stats, state_dict is pure weights
    )


def get_weights(model) -> list[np.ndarray]:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model, arrays) -> None:
    keys = model.state_dict().keys()
    model.load_state_dict(OrderedDict((k, torch.tensor(a)) for k, a in zip(keys, arrays)), strict=True)
