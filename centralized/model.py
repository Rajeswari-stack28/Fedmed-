import torch
import torch.nn as nn


class Simple3DUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder 1
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool3d(2)

        # Encoder 2
        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder 2
        self.up2 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder 1
        self.up1 = nn.ConvTranspose3d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Final output
        self.output = nn.Conv3d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool2(e2)
        )

        # Decoder
        d2 = self.up2(b)

        # Skip connection
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        # Skip connection
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)