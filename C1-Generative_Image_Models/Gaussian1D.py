import torch
import torch.nn as nn

class Gaussian1D(nn.Module):
    """
    A learnable 1D Gaussian density model.

    This module represents a univariate normal distribution with
    learnable mean (mu) and log-variance (log_var) parameters.
    The log-variance is optimized instead of the variance directly
    for numerical stability and to ensure positivity of the variance.

    The probability density function is:

        p(x) = (1 / sqrt(2πσ²)) * exp(-(x - μ)^2 / (2σ²))

    where:
        μ  = mean
        σ² = variance = exp(log_var)
    """

    def __init__(self):
        """
        Initialize the Gaussian1D model.

        The parameters are initialized as:
            mu = 0.0
            log_var = 0.0  (corresponds to variance = 1.0)
        """
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.log_var = nn.Parameter(torch.tensor(0.0))

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihood of input data under the model.

        Args:
            x (torch.Tensor):
                A 1D tensor of shape (N,) containing data samples.

        Returns:
            torch.Tensor:
                A tensor of shape (N,) containing the log-likelihood
                log p(x_i | μ, σ²) for each input sample.
        """
        var = torch.exp(self.log_var)

        # Note that log(1/sqrt(2 * torch.pi * var)) = -0.5 * log(2 * torch.pi * var)
        return (
            - 0.5 * torch.log(2 * torch.pi * var)
            - (x - self.mu) ** 2 / (2 * var)
        )

    def sample(self, n: int) -> torch.Tensor:
        """
        Draw samples from the learned Gaussian distribution.

        Args:
            n (int):
                Number of samples to generate.

        Returns:
            torch.Tensor:
                A tensor of shape (n,) containing samples drawn from
                N(μ, σ²), where σ² = exp(log_var).
        """
        var = torch.exp(self.log_var)
        return torch.randn(n) * torch.sqrt(var) + self.mu
