import torch
from torch.utils.data import Dataset


class SyntheticMRIDataset(Dataset):

    def __init__(self, num_samples=20):

        self.num_samples = num_samples

        # Create fake 3D MRI volumes
        self.images = torch.randn(
            num_samples,
            1,
            16,
            16,
            16
        )

        # Create tumor masks
        self.masks = torch.zeros(
            num_samples,
            1,
            16,
            16,
            16
        )

        # Create a small tumor region
        self.masks[
            :,
            :,
            6:10,
            6:10,
            6:10
        ] = 1.0

        # Make tumor region brighter
        self.images[
            :,
            :,
            6:10,
            6:10,
            6:10
        ] += 3.0

    def __len__(self):

        return self.num_samples

    def __getitem__(self, index):

        image = self.images[index]

        mask = self.masks[index]

        return image, mask