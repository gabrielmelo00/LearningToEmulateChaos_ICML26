import torch
import torch.nn as nn
import torch.nn.functional as F


class L63MLP(nn.Module):
    """
    3-layer MLP state-to-state map for Lorenz-63.

    Input shape:
        (B, 1, 3) or (B, 3)
    Output shape:
        (B, 1, 3) or (B, 3) matching the input rank.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        h = int(hidden_dim)
        self.fc1 = nn.Linear(3, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 3)

    def forward(self, x: torch.Tensor, param=None) -> torch.Tensor:
        if x.ndim == 3:
            b, t, d = x.shape
            x_flat = x.reshape(b * t, d)
            y = self.fc1(x_flat)
            y = F.gelu(y)
            y = self.fc2(y)
            y = F.gelu(y)
            y = self.fc3(y)
            return y.reshape(b, t, 3)

        if x.ndim == 2:
            y = self.fc1(x)
            y = F.gelu(y)
            y = self.fc2(y)
            y = F.gelu(y)
            y = self.fc3(y)
            return y

        raise ValueError(f"L63MLP expects input shape (B,1,3) or (B,3), got {tuple(x.shape)}")
