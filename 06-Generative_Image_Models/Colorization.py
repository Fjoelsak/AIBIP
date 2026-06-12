import os

import torch
import torch.nn as nn


class ColorizationEncoder(nn.Module):
    """
    Convolutional encoder that maps a single-channel (L) image to a latent feature map.

    Input is the L channel (lightness) of a Lab image, normalized to [0, 1].
    Three convolutional blocks with stride 2 progressively downsample the spatial
    dimensions while increasing channel depth.

    Args:
        base_channels (int): Number of channels in the first conv block. Default: 64.
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            # 1x32x32 -> base_channels x 16x16
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # base_channels x 16x16 -> base_channels*2 x 8x8
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # base_channels*2 x 8x8 -> base_channels*4 x 4x4
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): L channel images of shape (B, 1, 32, 32).

        Returns:
            torch.Tensor: Feature map of shape (B, base_channels*4, 4, 4).
        """
        return self.encoder(x)


class ColorizationDecoder(nn.Module):
    """
    Convolutional decoder that maps a latent feature map to the ab channels of a Lab image.

    Three transposed convolutional blocks upsample back to the original spatial resolution.
    The output is passed through Tanh and scaled to [-1, 1], matching the typical
    range of normalized ab channels.

    Args:
        base_channels (int): Must match the base_channels used in ColorizationEncoder. Default: 64.
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            # base_channels*4 x 4x4 -> base_channels*2 x 8x8
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # base_channels*2 x 8x8 -> base_channels x 16x16
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # base_channels x 16x16 -> 2 x 32x32
            nn.ConvTranspose2d(base_channels, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): Feature map of shape (B, base_channels*4, 4, 4).

        Returns:
            torch.Tensor: Predicted ab channels of shape (B, 2, 32, 32) in [-1, 1].
        """
        return self.decoder(z)


class ColorizationNet(nn.Module):
    """
    Self-supervised colorization network for CIFAR-10 images.

    Takes the L channel (lightness) of a Lab-converted image as input and predicts
    the ab channels (color). The network is trained with MSE loss against the
    ground-truth ab channels — no class labels are required.

    Args:
        base_channels (int): Base channel width. Default: 64.
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.encoder = ColorizationEncoder(base_channels)
        self.decoder = ColorizationDecoder(base_channels)

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            L (torch.Tensor): L channel images of shape (B, 1, 32, 32) in [0, 1].

        Returns:
            torch.Tensor: Predicted ab channels of shape (B, 2, 32, 32) in [-1, 1].
        """
        return self.decoder(self.encoder(L))

    def save_model(self, path: str = "models/colorization_cifar10.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path: str = "models/colorization_cifar10.pth", device: str = "cpu"):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            print(f"Model loaded from {path}.")
        else:
            print(f"File {path} not found.")
