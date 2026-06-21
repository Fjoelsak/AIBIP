import os

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Compact convolutional autoencoder for 28x28 grayscale images.

    Compresses a 1x28x28 image into a small ``latent_channels x 7 x 7`` feature
    map (a 16x spatial reduction) and reconstructs it. This is the perceptual
    compression stage of a latent diffusion model: diffusion later runs in the
    7x7 latent space instead of pixel space, which is far cheaper.

    The latent is intentionally low-dimensional but **spatial** (a small image,
    not a flat vector), so the diffusion U-Net can still exploit 2D structure.
    A Tanh on the latent keeps its range bounded, which stabilises the diffusion
    model trained on top of it.

    Args:
        latent_channels (int): Channel depth of the 7x7 latent. Default: 4.
        channels        (int): Base channel width. Default: 32.
    """

    def __init__(self, latent_channels: int = 4, channels: int = 32):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: 1x28x28 -> C x14x14 -> 2C x7x7 -> latent_channels x7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=4, stride=2, padding=1),       # 14x14
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, latent_channels, kernel_size=3, padding=1),     # 7x7
            nn.Tanh(),
        )

        # Decoder: latent_channels x7x7 -> 2C x7x7 -> C x14x14 -> 1x28x28
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels * 2, kernel_size=3, padding=1),     # 7x7
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(channels * 2, channels, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.ConvTranspose2d(channels, 1, kernel_size=4, stride=2, padding=1),    # 28x28
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to their latent representation.

        Args:
            x (torch.Tensor): Images of shape (B, 1, 28, 28) in [-1, 1].

        Returns:
            torch.Tensor: Latents of shape (B, latent_channels, 7, 7) in [-1, 1].
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to images.

        Args:
            z (torch.Tensor): Latents of shape (B, latent_channels, 7, 7).

        Returns:
            torch.Tensor: Reconstructed images of shape (B, 1, 28, 28) in [-1, 1].
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images by encoding and decoding.

        Args:
            x (torch.Tensor): Images of shape (B, 1, 28, 28) in [-1, 1].

        Returns:
            torch.Tensor: Reconstructions of shape (B, 1, 28, 28) in [-1, 1].
        """
        return self.decode(self.encode(x))

    def save_model(self, path: str = "models/autoencoder_fashion_mnist.pth"):
        """
        Save the full model state dictionary.

        Args:
            path (str): File path. Defaults to "models/autoencoder_fashion_mnist.pth".
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path: str = "models/autoencoder_fashion_mnist.pth",
                   device: str = "cpu"):
        """
        Load model state dictionary from file.

        Args:
            path   (str): File path. Defaults to "models/autoencoder_fashion_mnist.pth".
            device (str): Device to map parameters to. Default: "cpu".
        """
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            print(f"Model loaded from {path}.")
        else:
            print(f"File {path} not found.")
