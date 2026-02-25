import torch
import torch.nn as nn
import torch.nn.functional as F

import os

class MaskedConv2d(nn.Conv2d):
    """
    Two-dimensional masked convolution layer for autoregressive models.

    This layer extends ``torch.nn.Conv2d`` by applying a binary mask to the
    convolution weights. The mask ensures that each output pixel only depends
    on previously generated pixels, enforcing the autoregressive property
    required by PixelCNN.

    Two mask types are supported:

    - 'A': Excludes the current pixel (used in the first layer).
    - 'B': Includes the current pixel (used in subsequent layers).

    The mask is registered as a buffer and applied during the forward pass.
    """

    def __init__(self, mask_type, *args, **kwargs):
        """
        Initialize the masked convolution layer.

        Args:
            mask_type (str):
                Type of mask to apply. Must be either 'A' or 'B'.
            *args:
                Positional arguments passed to ``nn.Conv2d``.
            **kwargs:
                Keyword arguments passed to ``nn.Conv2d``.
        """
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.ones_like(self.weight))

        _, _, h, w = self.weight.shape

        # Zero out future pixels
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        """
        Apply masked convolution to the input.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor:
                Output tensor after masked convolution.
        """
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding
        )


class FastPixelCNN(nn.Module):
    """
    PixelCNN model for discrete grayscale image generation.

    This implementation models the joint distribution of image pixels
    autoregressively using masked convolutions. Each pixel is predicted
    conditioned on all previously generated pixels in raster-scan order.

    The model outputs logits for a categorical distribution over discrete
    pixel intensities.
    """

    def __init__(self, hidden_dims=128, num_layers=6):
        """
        Initialize the PixelCNN model.

        Args:
            hidden_dims (int, optional):
                Number of hidden feature channels used in masked convolution
                layers. Defaults to 128.
            num_layers (int, optional):
                Number of masked convolution layers of type 'B'.
                Defaults to 6.
        """
        super().__init__()

        self.num_classes = 32
        self.embed = nn.Embedding(self.num_classes, hidden_dims)

        layers = []

        # First layer (Type A mask)
        layers.append(MaskedConv2d('A', hidden_dims, hidden_dims, 5, padding=2))
        layers.append(nn.ReLU())

        # Subsequent layers (Type B mask)
        for _ in range(num_layers):
            layers.append(MaskedConv2d('B', hidden_dims, hidden_dims, 5, padding=2))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden_dims, self.num_classes, 1)

    def forward(self, x):
        """
        Perform a forward pass through the PixelCNN.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, 1, height, width)
                containing integer-encoded pixel values.

        Returns:
            torch.Tensor:
                Logits of shape (batch_size, num_classes, height, width)
                representing the categorical distribution over pixel values.
        """
        x = self.embed(x.squeeze(1)).permute(0, 3, 1, 2).float()
        x = self.net(x)
        return self.out(x)

    def save_model(self, path="models/deep_pixelcnn_mnist.pth"):
        """
        Save the model's state dictionary to a file.

        This method stores the current model parameters using PyTorch's
        ``torch.save`` function. Only the state dictionary is saved,
        not the entire model object.

        Args:
            path (str, optional): File path where the model state dictionary
                will be saved. Defaults to "deep_pixelcnn_mnist.pth".

        Returns:
            None
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, device, path="models/deep_pixelcnn_mnist.pth"):
        """
        Load the model's state dictionary from a file.

        This method loads model parameters from a file using PyTorch's
        ``torch.load`` function and moves the model to the specified device.
        If the file does not exist, a message is printed and no action is taken.

        Args:
            device (torch.device): The device on which the model should be loaded
                (e.g., CPU or CUDA device).
            path (str, optional): File path from which the model state dictionary
                will be loaded. Defaults to "deep_pixelcnn_mnist.pth".

        Returns:
            None
        """
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            print(f"Model loaded from {path}.")
        else:
            print(f"File {path} not found.")