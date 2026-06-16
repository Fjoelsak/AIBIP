import os

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Convolutional generator that maps a latent vector to a 28x28 grayscale image.

    A linear projection expands the latent vector to a spatial feature map,
    which is then upsampled by three transposed convolutional blocks. The final
    Tanh activation constrains pixel values to [-1, 1], matching the normalised
    input distribution expected by the discriminator.

    Args:
        latent_dim (int): Dimensionality of the input noise vector. Default: 64.
        channels   (int): Base channel width. Default: 64.
    """

    def __init__(self, latent_dim: int = 64, channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, channels * 4 * 4 * 4)
        self.net = nn.Sequential(
            # 256x4x4 -> 128x8x8
            nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            # 128x8x8 -> 64x16x16
            nn.ConvTranspose2d(channels * 2, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 64x16x16 -> 1x28x28  (kernel=4, stride=2, padding=3 gives exact 28)
            nn.ConvTranspose2d(channels, 1, kernel_size=4, stride=2, padding=3),
            nn.Tanh(),
        )
        self._channels = channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate images from a batch of latent vectors.

        Args:
            z (torch.Tensor): Latent vectors of shape (B, latent_dim).

        Returns:
            torch.Tensor: Generated images of shape (B, 1, 28, 28) in [-1, 1].
        """
        x = self.fc(z).view(-1, self._channels * 4, 4, 4)
        return self.net(x)


class Discriminator(nn.Module):
    """
    Convolutional discriminator that classifies images as real or generated.

    Three convolutional blocks (Conv2d -> BatchNorm2d -> LeakyReLU) with
    stride 2 progressively reduce the spatial dimensions. A final linear
    layer with Sigmoid outputs the probability of the input being real.

    LeakyReLU (negative slope 0.2) is standard in GANs: it avoids dead
    neurons in the discriminator without the mode-collapse risk of ReLU.

    Args:
        channels (int): Base channel width. Default: 64.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # 1x28x28 -> 64x14x14
            nn.Conv2d(1, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x14x14 -> 128x7x7
            nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x7x7 -> 256x3x3
            nn.Conv2d(channels * 2, channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(channels * 4 * 3 * 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of images as real (→ 1) or generated (→ 0).

        Args:
            x (torch.Tensor): Images of shape (B, 1, 28, 28) in [-1, 1].

        Returns:
            torch.Tensor: Probabilities of shape (B, 1).
        """
        h = self.net(x).flatten(start_dim=1)
        return torch.sigmoid(self.fc(h))


class GAN(nn.Module):
    """
    Generative Adversarial Network (GAN) for image synthesis.

    Wraps a Generator and a Discriminator trained via the minimax objective
    (Goodfellow et al., 2014):

        min_G max_D  E[log D(x)]  +  E[log(1 - D(G(z)))]

    In practice the generator minimises -E[log D(G(z))] (non-saturating loss)
    for better gradient flow early in training.

    Training alternates between:
      1. Updating D to better distinguish real from generated images.
      2. Updating G to fool D into classifying generated images as real.

    Args:
        latent_dim (int): Dimensionality of the generator's noise input. Default: 64.
        channels   (int): Base channel width for both G and D. Default: 64.
    """

    def __init__(self, latent_dim: int = 64, channels: int = 64):
        super().__init__()
        self.latent_dim    = latent_dim
        self.generator     = Generator(latent_dim, channels)
        self.discriminator = Discriminator(channels)

    def generate(self, n: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample n images from the generator by drawing noise from N(0, I).

        Args:
            n      (int): Number of images to generate.
            device (str): Target device. Default: "cpu".

        Returns:
            torch.Tensor: Generated images of shape (n, 1, 28, 28) in [-1, 1].
        """
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.generator(z)

    def save_model(self, path: str = "models/gan_fashion_mnist.pth"):
        """
        Save the full model state dictionary.

        Args:
            path (str): File path. Defaults to "models/gan_fashion_mnist.pth".
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path: str = "models/gan_fashion_mnist.pth", device: str = "cpu"):
        """
        Load model state dictionary from file.

        Args:
            path   (str): File path. Defaults to "models/gan_fashion_mnist.pth".
            device (str): Device to map parameters to. Default: "cpu".
        """
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            print(f"Model loaded from {path}.")
        else:
            print(f"File {path} not found.")
