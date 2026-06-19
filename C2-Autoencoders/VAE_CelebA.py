from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


class Encoder(nn.Module):
    """
    Convolutional encoder that maps a 64x64 RGB image to the parameters of a
    Gaussian posterior q(z|x): mean mu and log-variance log_var.

    Four convolutional blocks (Conv2d -> BatchNorm2d -> ReLU) with stride 2
    progressively downsample the spatial dimensions from 64x64 to 4x4 while
    increasing the channel count. Two separate linear projection heads produce
    mu and log_var independently.

    Args:
        latent_dim (int): Dimensionality of the latent space. Default: 256.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            # 3x64x64 -> 32x32x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x32x32 -> 64x16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x16x16 -> 128x8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128x8x8 -> 256x4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an image batch to posterior parameters.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, 64, 64) in [0, 1].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                mu      — posterior mean,     shape (B, latent_dim)
                log_var — posterior log-var,  shape (B, latent_dim)
        """
        h = self.conv(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """
    Convolutional decoder that maps a latent vector back to a 64x64 RGB image.

    A linear projection reshapes the latent vector to a 256x4x4 feature map,
    followed by four transposed convolutional blocks that upsample back to
    64x64 and a final sigmoid activation constraining pixels to [0, 1].

    Args:
        latent_dim (int): Dimensionality of the latent space. Default: 256.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            # 256x4x4 -> 128x8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128x8x8 -> 64x16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x16x16 -> 32x32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x32x32 -> 3x64x64
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent vectors to images.

        Args:
            z (torch.Tensor): Latent vectors of shape (B, latent_dim).

        Returns:
            torch.Tensor: Reconstructed images of shape (B, 3, 64, 64) in [0, 1].
        """
        x = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(x)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for generative modelling of CelebA faces.

    This is the RGB 64x64 counterpart of the grayscale MNIST VAE in ``VAE.py``.
    The encoder parametrises a Gaussian posterior q(z|x) = N(mu, sigma^2 I);
    a sample z is drawn via the reparametrisation trick, and the decoder maps
    z back to image space.

    The training objective is the ELBO (Evidence Lower BOund):

        ELBO = E_q[log p(x|z)]  -  KL( q(z|x) || p(z) )
             = -Reconstruction loss  -  KL divergence

    where p(z) = N(0, I) is the standard normal prior.

    Args:
        latent_dim (int): Dimensionality of the latent space. Default: 256.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Draw a latent sample using the reparametrisation trick.

        Instead of sampling z ~ N(mu, sigma^2) directly (non-differentiable),
        we write  z = mu + sigma * eps  with  eps ~ N(0, I),  which keeps the
        gradient path through mu and log_var intact.

        Args:
            mu      (torch.Tensor): Posterior mean,     shape (B, latent_dim).
            log_var (torch.Tensor): Posterior log-var,  shape (B, latent_dim).

        Returns:
            torch.Tensor: Latent samples of shape (B, latent_dim).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode, sample, and decode an image batch.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, 64, 64) in [0, 1].

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                x_hat   — reconstructed images, shape (B, 3, 64, 64)
                mu      — posterior mean,        shape (B, latent_dim)
                log_var — posterior log-var,     shape (B, latent_dim)
        """
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        return self.decoder(z), mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the posterior mean mu as a deterministic latent code.

        Using mu (rather than a random sample) is standard practice for
        visualisation, interpolation, and attribute arithmetic — it gives the
        most likely z without noise.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, 64, 64) in [0, 1].

        Returns:
            torch.Tensor: Posterior means of shape (B, latent_dim).
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent vectors to images.

        Args:
            z (torch.Tensor): Latent vectors of shape (B, latent_dim).

        Returns:
            torch.Tensor: Generated images of shape (B, 3, 64, 64) in [0, 1].
        """
        return self.decoder(z)

    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate n new face images by sampling from the prior p(z) = N(0, I).

        Args:
            n      (int): Number of images to generate.
            device (str): Target device. Default: "cpu".

        Returns:
            torch.Tensor: Generated images of shape (n, 3, 64, 64) in [0, 1].
        """
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decoder(z)

    def interpolate(self, x_a: torch.Tensor, x_b: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Linearly interpolate in latent space between two images.

        Both inputs are encoded to their posterior means and the straight line
        between them is decoded at ``steps`` equally spaced points, producing a
        smooth morph from x_a to x_b.

        Args:
            x_a   (torch.Tensor): Start image of shape (1, 3, 64, 64) or (3, 64, 64).
            x_b   (torch.Tensor): End image, same shape as x_a.
            steps (int):          Number of interpolation points. Default: 10.

        Returns:
            torch.Tensor: Decoded images of shape (steps, 3, 64, 64) in [0, 1].
        """
        x_a = x_a.unsqueeze(0) if x_a.dim() == 3 else x_a
        x_b = x_b.unsqueeze(0) if x_b.dim() == 3 else x_b
        with torch.no_grad():
            z_a = self.encode(x_a)
            z_b = self.encode(x_b)
            alphas = torch.linspace(0.0, 1.0, steps, device=z_a.device).view(-1, 1)
            z = (1.0 - alphas) * z_a + alphas * z_b
            return self.decoder(z)

    def save_model(self, path: str = "models/vae_celeba.pth"):
        """
        Save the model's state dictionary to a file.

        Args:
            path (str): File path for the saved state dictionary.
                Defaults to "models/vae_celeba.pth".
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(
        self,
        path: str = "models/vae_celeba.pth",
        url: str | None = None,
        device: str = "cpu",
    ):
        """
        Load the model's state dictionary, preferring a local file over a URL.

        Loading order:
            1. If ``path`` exists locally, load from there.
            2. Otherwise, if ``url`` is given, download the weights via
               ``torch.hub.load_state_dict_from_url`` (cached by torch.hub).
            3. Otherwise, report that no weights were found.

        This lets the same notebook run both locally (weights committed to
        ``models/``) and on a fresh Colab runtime (weights pulled from a URL).

        Args:
            path   (str):          Local file path of the saved state dictionary.
                Defaults to "models/vae_celeba.pth".
            url    (str | None):   Optional URL to download the weights from if
                the local file is absent. Default: None.
            device (str):          Device to map the loaded parameters to.
                Default: "cpu".
        """
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=device)
            self.load_state_dict(state_dict)
            self.to(device)
            print(f"Model loaded from {path}.")
        elif url is not None:
            state_dict = load_state_dict_from_url(url, map_location=device)
            self.load_state_dict(state_dict)
            self.to(device)
            print(f"Model loaded from {url}.")
        else:
            print(f"File {path} not found and no URL provided.")


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the negative ELBO (the quantity minimised during training).

    The reconstruction term is the per-image summed binary cross-entropy
    between input and reconstruction; the regulariser is the closed-form KL
    divergence between the Gaussian posterior and the standard normal prior.
    Both terms are averaged over the batch.

    Args:
        x       (torch.Tensor): Ground-truth images, shape (B, 3, 64, 64) in [0, 1].
        x_hat   (torch.Tensor): Reconstructed images, same shape as x.
        mu      (torch.Tensor): Posterior mean,    shape (B, latent_dim).
        log_var (torch.Tensor): Posterior log-var, shape (B, latent_dim).
        beta    (float):        Weight on the KL term (beta-VAE). Default: 1.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            loss     — total negative ELBO (scalar)
            recon    — reconstruction term  (scalar)
            kl       — KL divergence term   (scalar)
    """
    batch_size = x.size(0)
    recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    return recon + beta * kl, recon, kl


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Closed-form KL divergence KL(q(z|x) || N(0, I)), averaged over the batch.

    Args:
        mu      (torch.Tensor): Posterior mean,    shape (B, latent_dim).
        log_var (torch.Tensor): Posterior log-var, shape (B, latent_dim).

    Returns:
        torch.Tensor: Scalar KL divergence.
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)


class PerceptualLoss(nn.Module):
    """
    VGG16 feature-space reconstruction loss for sharper image generation.

    Plain pixel losses (BCE/MSE) penalise per-pixel intensity differences,
    which encourages the decoder to output the blurry per-pixel average of all
    plausible reconstructions. Comparing images in the feature space of a
    pre-trained VGG16 network instead rewards matching edges, textures, and
    structure, producing visibly sharper faces.

    The total reconstruction term combes a pixel MSE with the VGG feature MSE:

        recon = MSE(x_hat, x)  +  perceptual_weight * sum_l MSE(phi_l(x_hat), phi_l(x))

    where ``phi_l`` are the activations at a few selected VGG layers. The VGG
    weights are frozen and excluded from optimisation.

    Args:
        perceptual_weight (float): Weight on the VGG feature term. Default: 0.1.
        layers (tuple[int, ...]):  Indices into ``vgg16.features`` at whose
            outputs the feature distance is measured. Defaults to relu1_2,
            relu2_2, relu3_3.
    """

    # ImageNet normalisation constants expected by torchvision VGG.
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, perceptual_weight: float = 0.1, layers: tuple[int, ...] = (3, 8, 15)):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.perceptual_weight = perceptual_weight
        self.layers = layers
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run x through VGG and collect activations at the selected layers."""
        x = (x - self.mean) / self.std
        feats, h = [], x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layers:
                feats.append(h)
            if i >= max(self.layers):
                break
        return feats

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel MSE plus weighted VGG feature MSE between x_hat and x.

        Args:
            x_hat (torch.Tensor): Reconstructed images, shape (B, 3, 64, 64) in [0, 1].
            x     (torch.Tensor): Ground-truth images, same shape as x_hat.

        Returns:
            torch.Tensor: Scalar reconstruction loss.
        """
        pixel = nn.functional.mse_loss(x_hat, x)
        feat = sum(
            nn.functional.mse_loss(fh, ft)
            for fh, ft in zip(self._features(x_hat), self._features(x))
        )
        return pixel + self.perceptual_weight * feat
