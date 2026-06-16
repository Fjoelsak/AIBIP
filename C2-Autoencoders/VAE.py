import os

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional encoder that maps a 28x28 grayscale image to the parameters
    of a Gaussian posterior q(z|x): mean mu and log-variance log_var.

    Three convolutional blocks (Conv2d -> BatchNorm2d -> ReLU) with stride 2
    progressively downsample the spatial dimensions. Two separate linear
    projection heads produce mu and log_var independently.

    Args:
        latent_dim (int): Dimensionality of the latent space. Default: 2.
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
        self.fc_mu      = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an image batch to posterior parameters.

        Args:
            x (torch.Tensor): Input images of shape (B, 1, 28, 28) in [0, 1].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                mu      — posterior mean,     shape (B, latent_dim)
                log_var — posterior log-var,  shape (B, latent_dim)
        """
        h = self.conv(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """
    Convolutional decoder that maps a latent vector back to a 28x28 image.

    Identical to the Autoencoder decoder: a linear projection followed by
    three transposed convolutional blocks and a final sigmoid activation.

    Args:
        latent_dim (int): Dimensionality of the latent space. Default: 2.
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
        x = self.fc(z).view(-1, 128, 4, 4)
        return self.deconv(x)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for generative modelling on MNIST.

    Unlike a plain autoencoder, the encoder does not produce a single latent
    point but instead parametrises a Gaussian posterior q(z|x) = N(mu, sigma^2 I).
    A sample z is drawn via the reparametrisation trick, making the gradient
    flow through the stochastic node possible.

    The training objective is the ELBO (Evidence Lower BOund):

        ELBO = E_q[log p(x|z)]  -  KL( q(z|x) || p(z) )
             = -Reconstruction loss  -  KL divergence

    where p(z) = N(0, I) is the standard normal prior.

    Args:
        latent_dim (int): Dimensionality of the latent space. Default: 2.
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Draw a latent sample using the reparametrisation trick.

        Instead of sampling z ~ N(mu, sigma^2) directly (non-differentiable),
        we write  z = mu + sigma * eps  with  eps ~ N(0, I),  which keeps
        the gradient path through mu and log_var intact.

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
            x (torch.Tensor): Input images of shape (B, 1, 28, 28) in [0, 1].

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                x_hat   — reconstructed images, shape (B, 1, 28, 28)
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
        visualisation and downstream tasks — it gives the most likely z
        without noise.

        Args:
            x (torch.Tensor): Input images of shape (B, 1, 28, 28) in [0, 1].

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
            torch.Tensor: Generated images of shape (B, 1, 28, 28) in [0, 1].
        """
        return self.decoder(z)

    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate n new images by sampling from the prior p(z) = N(0, I).

        Args:
            n      (int): Number of images to generate.
            device (str): Target device. Default: "cpu".

        Returns:
            torch.Tensor: Generated images of shape (n, 1, 28, 28) in [0, 1].
        """
        z = torch.randn(n, self.decoder.fc.in_features, device=device)
        with torch.no_grad():
            return self.decoder(z)

    def save_model(self, path: str = "models/vae_mnist.pth"):
        """
        Save the model's state dictionary to a file.

        Args:
            path (str): File path for the saved state dictionary.
                Defaults to "models/vae_mnist.pth".
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path: str = "models/vae_mnist.pth", device: str = "cpu"):
        """
        Load the model's state dictionary from a file.

        Args:
            path   (str): File path of the saved state dictionary.
                Defaults to "models/vae_mnist.pth".
            device (str): Device to map the loaded parameters to. Default: "cpu".
        """
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            print(f"Model loaded from {path}.")
        else:
            print(f"File {path} not found.")
