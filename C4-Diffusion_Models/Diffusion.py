import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Build sinusoidal timestep embeddings (as in the Transformer / DDPM papers).

    Each timestep is mapped to a `dim`-dimensional vector of sines and cosines
    at geometrically spaced frequencies. This gives the network a smooth,
    continuous notion of "how noisy" the current input is.

    Args:
        timesteps (torch.Tensor): Integer timesteps of shape (B,).
        dim       (int): Output embedding dimension (should be even).

    Returns:
        torch.Tensor: Embeddings of shape (B, dim).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1)
    )
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # zero-pad if dim is odd
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    """
    Convolutional residual block with timestep conditioning.

    Two 3x3 convolutions (GroupNorm + SiLU) with a skip connection. The
    timestep embedding is projected and added to the feature maps between
    the two convolutions, so every block "knows" the current noise level.

    Args:
        in_ch    (int): Number of input channels.
        out_ch   (int): Number of output channels.
        time_dim (int): Dimensionality of the timestep embedding.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # 1x1 conv to match channel counts on the skip path if needed
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Inject timestep information (broadcast over spatial dimensions)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """
    Compact U-Net that predicts the noise added to an image at timestep t.

    A three-level encoder/decoder with skip connections, sized for 28x28
    grayscale images (Fashion-MNIST / MNIST). The network takes the noisy
    image x_t and the timestep t, and outputs an estimate of the noise eps.

    Args:
        channels  (int): Base channel width. Default: 64.
        time_dim  (int): Timestep embedding dimension. Default: 256.
    """

    def __init__(self, channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.in_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)        # 28x28
        self.down1 = ResidualBlock(channels, channels, time_dim)
        self.pool1 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)        # 14x14
        self.down2 = ResidualBlock(channels, channels * 2, time_dim)
        self.pool2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=4, stride=2, padding=1)  # 7x7

        # Bottleneck
        self.mid = ResidualBlock(channels * 2, channels * 2, time_dim)

        # Decoder (channels double after concatenation with skip features)
        self.up2 = nn.ConvTranspose2d(channels * 2, channels * 2, kernel_size=4, stride=2, padding=1)  # 14x14
        self.dec2 = ResidualBlock(channels * 4, channels, time_dim)
        self.up1 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)          # 28x28
        self.dec1 = ResidualBlock(channels * 2, channels, time_dim)

        self.out_norm = nn.GroupNorm(8, channels)
        self.out_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise present in x at timestep t.

        Args:
            x (torch.Tensor): Noisy images of shape (B, 1, 28, 28).
            t (torch.Tensor): Integer timesteps of shape (B,).

        Returns:
            torch.Tensor: Predicted noise of shape (B, 1, 28, 28).
        """
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)        # 28x28, C
        d2 = self.down2(self.pool1(d1), t_emb)  # 14x14, 2C
        m = self.mid(self.pool2(d2), t_emb)     # 7x7,  2C

        u2 = self.up2(m)                  # 14x14, 2C
        u2 = self.dec2(torch.cat([u2, d2], dim=1), t_emb)  # -> C
        u1 = self.up1(u2)                 # 28x28, C
        u1 = self.dec1(torch.cat([u1, d1], dim=1), t_emb)  # -> C

        return self.out_conv(F.silu(self.out_norm(u1)))


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020).

    Wraps a noise-prediction U-Net together with a fixed linear variance
    schedule. The forward process gradually corrupts an image with Gaussian
    noise over T steps; the model learns to reverse it one step at a time.

    Forward process (closed form):
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    Training objective (simplified, Ho et al.):
        L = E_{x_0, t, eps} || eps - eps_theta(x_t, t) ||^2

    Args:
        timesteps  (int): Number of diffusion steps T. Default: 1000.
        beta_start (float): First (smallest) noise variance. Default: 1e-4.
        beta_end   (float): Last (largest) noise variance. Default: 0.02.
        channels   (int): Base channel width of the U-Net. Default: 64.
    """

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, channels: int = 64):
        super().__init__()
        self.timesteps = timesteps
        self.model = UNet(channels=channels)

        # Linear variance schedule and the derived quantities used by the
        # closed-form forward process and the reverse sampling step.
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # Registered as buffers so they move with .to(device) and are saved.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        Sample x_t from the forward process q(x_t | x_0) in closed form.

        Args:
            x0    (torch.Tensor): Clean images of shape (B, 1, 28, 28).
            t     (torch.Tensor): Integer timesteps of shape (B,).
            noise (torch.Tensor): Optional pre-sampled noise. If None, drawn
                                  from N(0, I).

        Returns:
            torch.Tensor: Noisy images x_t of shape (B, 1, 28, 28).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alphas_bar[t][:, None, None, None]
        sb = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]
        return sa * x0 + sb * noise

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute the simplified DDPM training loss for a batch of images.

        A random timestep is drawn per image, noise is added via q_sample,
        and the model is trained to predict that noise (MSE).

        Args:
            x0 (torch.Tensor): Clean images of shape (B, 1, 28, 28) in [-1, 1].

        Returns:
            torch.Tensor: Scalar MSE loss.
        """
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, n: int, device: str = "cpu",
               return_trajectory: bool = False):
        """
        Generate images by running the full reverse process from pure noise.

        Starting from x_T ~ N(0, I), the model iteratively denoises down to
        x_0 using the ancestral sampling rule of Ho et al. (2020).

        Args:
            n                 (int): Number of images to generate.
            device            (str): Target device. Default: "cpu".
            return_trajectory (bool): If True, also return a list of
                                      intermediate samples for visualisation.

        Returns:
            torch.Tensor: Generated images of shape (n, 1, 28, 28) in [-1, 1].
            (optionally) list[torch.Tensor]: Intermediate states if requested.
        """
        x = torch.randn(n, 1, 28, 28, device=device)
        trajectory = []

        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            noise_pred = self.model(x, t)

            alpha = self.alphas[step]
            alpha_bar = self.alphas_bar[step]
            beta = self.betas[step]

            # Posterior mean of x_{t-1} given x_t and the predicted noise.
            coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            mean = (x - coef * noise_pred) / torch.sqrt(alpha)

            if step > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean  # no noise added at the final step

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    def save_model(self, path: str = "models/ddpm_fashion_mnist.pth"):
        """
        Save the full model state dictionary.

        Args:
            path (str): File path. Defaults to "models/ddpm_fashion_mnist.pth".
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path: str = "models/ddpm_fashion_mnist.pth", device: str = "cpu"):
        """
        Load model state dictionary from file.

        Args:
            path   (str): File path. Defaults to "models/ddpm_fashion_mnist.pth".
            device (str): Device to map parameters to. Default: "cpu".
        """
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            print(f"Model loaded from {path}.")
        else:
            print(f"File {path} not found.")
