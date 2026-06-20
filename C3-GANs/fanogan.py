"""
f-AnoGAN model definitions for image anomaly detection (RGB, 128x128).

f-AnoGAN (Schlegl et al., 2019, "f-AnoGAN: Fast unsupervised anomaly detection
with generative adversarial networks") extends AnoGAN by adding an encoder, so
that mapping a query image to its latent code takes a single forward pass
instead of an expensive per-image latent optimisation.

The method has three components, trained on **normal (defect-free) images only**:

1. Generator  G: z -> image            (learns the manifold of normal images)
2. Discriminator D: image -> score     (WGAN critic; also a feature extractor)
3. Encoder    E: image -> z            (inverts G, trained with G and D frozen)

At test time the anomaly score of an image x combines an image-space residual
and a discriminator-feature residual:

    A(x) = ||x - G(E(x))||^2  +  kappa * ||f(x) - f(G(E(x)))||^2

where f(.) are intermediate discriminator features. Normal images can be
reconstructed well by G(E(x)) and yield a low score; anomalies lie off the
learned manifold, are reconstructed poorly, and yield a high score. The
per-pixel residual |x - G(E(x))| additionally localises the anomaly.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN-style generator mapping a latent vector to a 128x128 RGB image.

    Five transposed-convolution blocks upsample a 1x1 latent map to 128x128.
    The final Tanh maps outputs to [-1, 1], so inputs must be normalised the
    same way.

    Args:
        latent_dim (int): Dimensionality of the latent space. Default: 128.
        channels   (int): Base channel count. Default: 64.
    """

    def __init__(self, latent_dim: int = 128, channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        c = channels
        self.net = nn.Sequential(
            # latent_dim x 1 x 1 -> (c*16) x 4 x 4
            nn.ConvTranspose2d(latent_dim, c * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(c * 16), nn.ReLU(True),
            # -> (c*8) x 8 x 8
            nn.ConvTranspose2d(c * 16, c * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 8), nn.ReLU(True),
            # -> (c*4) x 16 x 16
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 4), nn.ReLU(True),
            # -> (c*2) x 32 x 32
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2), nn.ReLU(True),
            # -> c x 64 x 64
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(True),
            # -> 3 x 128 x 128
            nn.ConvTranspose2d(c, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): Latent vectors of shape (B, latent_dim) or
                (B, latent_dim, 1, 1).

        Returns:
            torch.Tensor: Images of shape (B, 3, 128, 128) in [-1, 1].
        """
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """
    WGAN critic for 128x128 RGB images, doubling as a feature extractor.

    ``forward`` returns the scalar critic score; ``features`` returns the
    activations of the penultimate convolutional block, used for the
    feature-residual term of the f-AnoGAN anomaly score.

    Args:
        channels (int): Base channel count. Default: 64.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        c = channels
        self.features_net = nn.Sequential(
            # 3 x 128 x 128 -> c x 64 x 64
            nn.Conv2d(3, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            # -> (c*2) x 32 x 32
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.InstanceNorm2d(c * 2, affine=True), nn.LeakyReLU(0.2, inplace=True),
            # -> (c*4) x 16 x 16
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.InstanceNorm2d(c * 4, affine=True), nn.LeakyReLU(0.2, inplace=True),
            # -> (c*8) x 8 x 8
            nn.Conv2d(c * 4, c * 8, 4, 2, 1),
            nn.InstanceNorm2d(c * 8, affine=True), nn.LeakyReLU(0.2, inplace=True),
            # -> (c*16) x 4 x 4   (these are the f-AnoGAN features)
            nn.Conv2d(c * 8, c * 16, 4, 2, 1),
            nn.InstanceNorm2d(c * 16, affine=True), nn.LeakyReLU(0.2, inplace=True),
        )
        self.critic = nn.Conv2d(c * 16, 1, 4, 1, 0)  # -> 1 x 1 x 1

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate-block feature map (B, c*16, 4, 4)."""
        return self.features_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Images of shape (B, 3, 128, 128) in [-1, 1].

        Returns:
            torch.Tensor: Critic scores of shape (B,).
        """
        return self.critic(self.features_net(x)).view(-1)


class Encoder(nn.Module):
    """
    Encoder mapping a 128x128 RGB image to a latent vector (inverts G).

    Mirrors the discriminator's downsampling path and projects to ``latent_dim``
    with a final Tanh, matching the bounded latent range used during the
    izi-style encoder training.

    Args:
        latent_dim (int): Dimensionality of the latent space. Default: 128.
        channels   (int): Base channel count. Default: 64.
    """

    def __init__(self, latent_dim: int = 128, channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        c = channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1),
            nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, c * 16, 4, 2, 1),
            nn.BatchNorm2d(c * 16), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 16, latent_dim, 4, 1, 0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Images of shape (B, 3, 128, 128) in [-1, 1].

        Returns:
            torch.Tensor: Latent vectors of shape (B, latent_dim).
        """
        return self.net(x).view(x.size(0), self.latent_dim)


def anomaly_score(
    x: torch.Tensor,
    generator: Generator,
    discriminator: Discriminator,
    encoder: Encoder,
    kappa: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the f-AnoGAN anomaly score and per-pixel residual map.

    Score per image:
        A(x) = mean(|x - x_rec|^2)  +  kappa * mean(|f(x) - f(x_rec)|^2)
    with x_rec = G(E(x)) and f(.) the discriminator features.

    Args:
        x             (torch.Tensor): Images (B, 3, 128, 128) in [-1, 1].
        generator     (Generator):     Trained generator G.
        discriminator (Discriminator): Trained discriminator D.
        encoder       (Encoder):       Trained encoder E.
        kappa         (float):         Weight of the feature-residual term.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            scores   — anomaly score per image, shape (B,)
            residual — per-pixel residual map |x - x_rec| averaged over
                       channels, shape (B, 1, 128, 128), for localisation.
    """
    generator.eval(); discriminator.eval(); encoder.eval()
    with torch.no_grad():
        z = encoder(x)
        x_rec = generator(z)
        img_res = (x - x_rec).pow(2)
        feat_res = (discriminator.features(x) - discriminator.features(x_rec)).pow(2)
        scores = img_res.flatten(1).mean(dim=1) + kappa * feat_res.flatten(1).mean(dim=1)
        residual = (x - x_rec).abs().mean(dim=1, keepdim=True)
    return scores, residual


def _save(module: nn.Module, path: str) -> None:
    """Save a module's state dict, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(module.state_dict(), path)
    print(f"Saved to {path}.")


def _load(module: nn.Module, path: str, device: str = "cpu") -> None:
    """Load a module's state dict if the file exists, else report it."""
    if os.path.exists(path):
        module.load_state_dict(torch.load(path, map_location=device))
        module.to(device)
        print(f"Loaded from {path}.")
    else:
        print(f"File {path} not found.")
