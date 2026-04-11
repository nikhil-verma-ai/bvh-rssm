"""
Shared network utilities for BVH-RSSM.

All network modules build on MLP and LayerNormMLP. No other shared utilities —
keep this file focused on building blocks only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """Multi-layer perceptron with SiLU activations.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        hidden_dim: Width of hidden layers.
        n_layers: Number of hidden layers (0 = linear projection only).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 0:
            raise ValueError(f"n_layers must be >= 0, got {n_layers}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev_dim, hidden_dim), nn.SiLU()]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LayerNormMLP(nn.Module):
    """MLP with LayerNorm after each hidden layer (used in RSSM encoder/decoder).

    LayerNorm stabilizes training when processing the concatenated latent
    [h_t; z_t] which mixes deterministic and stochastic components.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        hidden_dim: Width of hidden layers.
        n_layers: Number of hidden layers.

    Note: n_layers=0 produces a bare linear layer with no normalization.
          hidden_dim is ignored when n_layers=0.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 0:
            raise ValueError(f"n_layers must be >= 0, got {n_layers}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
