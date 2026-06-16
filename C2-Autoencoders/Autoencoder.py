import os

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional encoder that maps a 28x28 grayscale image to a latent vector.

    Three convolutional blocks (Conv2d → BatchNorm2d → ReLU) with stride 2
    progressively downsample the spatial dimensions. A final linear layer
    projects the flattened feature map to the latent space.

    Args:
        latent_dim (int): Dimensionality of the latent vector. Default: 2.
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            # 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x7x7 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image batch to latent vectors.

        Args:
            x (torch.Tensor): Input images of shape (B, 1, 28, 28) in [0, 1].

        Returns:
            torch.Tensor: Latent vectors of shape (B, latent_dim).
        """
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class Decoder(nn.Module):
    """
    Convolutional decoder that maps a latent vector back to a 28x28 image.

    A linear layer projects the latent vector to a spatial feature map, which
    is then upsampled by three transposed convolutional blocks. A final sigmoid
    activation constrains pixel values to [0, 1].

    Args:
        latent_dim (int): Dimensionality of the latent vector. Default: 2.
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            # 128x4x4 -> 64x8x8
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x8x8 -> 32x15x15
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x15x15 -> 1x28x28
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=2, output_padding=0),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent vectors to images.

        Args:
            z (torch.Tensor): Latent vectors of shape (B, latent_dim).

        Returns:
            torch.Tensor: Reconstructed images of shape (B, 1, 28, 28) in [0, 1].
        """
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        return self.deconv(x)


class Autoencoder(nn.Module):
    """
    Convolutional autoencoder for unsupervised representation learning on MNIST.

    Combines an Encoder and a Decoder. The model is trained to minimize the
    mean squared error between input images and their reconstructions, forcing
    the network to learn a compact latent representation.

    Args:
        latent_dim (int): Dimensionality of the latent space. Default: 2.
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct an image batch via encode → decode.

        Args:
            x (torch.Tensor): Input images of shape (B, 1, 28, 28) in [0, 1].

        Returns:
            torch.Tensor: Reconstructed images of shape (B, 1, 28, 28) in [0, 1].
        """
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors.

        Args:
            x (torch.Tensor): Input images of shape (B, 1, 28, 28) in [0, 1].

        Returns:
            torch.Tensor: Latent vectors of shape (B, latent_dim).
        """
        return self.encoder(x)

    def save_model(self, path: str = "models/autoencoder_mnist.pth"):
        """
        Save the model's state dictionary to a file.

        Args:
            path (str): File path for the saved state dictionary.
                Defaults to "models/autoencoder_mnist.pth".
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path: str = "models/autoencoder_mnist.pth", device: str = "cpu"):
        """
        Load the model's state dictionary from a file.

        Args:
            path (str): File path of the saved state dictionary.
                Defaults to "models/autoencoder_mnist.pth".
            device (str): Device to map the loaded parameters to. Default: "cpu".
        """
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            print(f"Model loaded from {path}.")
        else:
            print(f"File {path} not found.")
