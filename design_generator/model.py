"""DesignGeneratorNet: 2-layer feedforward neural network for design generation.

This module defines the core PyTorch model architecture that maps design embeddings
(128-dim) to component vectors (512-dim) for generative design specifications.
"""

import torch
import torch.nn as nn


class DesignGeneratorNet(nn.Module):
    """2-layer feedforward network: Linear → ReLU → Linear.

    Maps design embeddings (128 dimensions) to component vectors (512 dimensions).
    This enables learning a transformation from compressed design representations
    to detailed rendering specifications.

    Architecture:
        fc1: Linear(input_dim → hidden_dim)
        relu: ReLU activation
        fc2: Linear(hidden_dim → output_dim)

    Forward pass:
        input (batch_size, 128) → fc1 → (batch_size, 256) → relu → fc2 → (batch_size, 512)

    Example:
        >>> model = DesignGeneratorNet(input_dim=128, hidden_dim=256, output_dim=512)
        >>> x = torch.randn(16, 128)  # batch of 16 designs
        >>> output = model(x)
        >>> output.shape
        torch.Size([16, 512])
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
    ) -> None:
        """Initialize the 2-layer network.

        Args:
            input_dim: Dimension of input design embeddings (default: 128)
            hidden_dim: Dimension of hidden layer (default: 256)
            output_dim: Dimension of output component vectors (default: 512)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim), typically (batch_size, 128)

        Returns:
            Output tensor of shape (batch_size, output_dim), typically (batch_size, 512)
            with dtype matching input tensor (typically float32)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
