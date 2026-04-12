"""
Actor and Critic networks for BVH-RSSM.

Both take the concatenated RSSM latent [h_t; z_t] as input.
Training logic lives in Plan 5 (training/). These are pure forward-pass
network definitions.

Actor: outputs action distribution parameters.
  - Continuous: (mean, log_std) for Gaussian policy
  - Discrete: logits for categorical policy

Critic: outputs value distribution (twohot over n_bins).
  Value head mirrors RewardHead design — categorical cross-entropy loss
  with symlog-encoded targets. Full training in Plan 5.
"""
from __future__ import annotations

from typing import Tuple, Union

import torch.nn as nn
from torch import Tensor

from bvh_rssm.networks.common import LayerNormMLP

_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


class Actor(nn.Module):
    """Policy network for the actor-critic.

    Args:
        latent_dim: Dimension of [h_t; z_t].
        action_dim: Number of action dimensions (continuous) or classes (discrete).
        discrete: If True, output logits for categorical policy.
                  If False, output (mean, log_std) for Gaussian policy.
        hidden_dim: MLP width.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        discrete: bool = False,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        out_dim = action_dim if discrete else action_dim * 2  # mean + log_std
        self.mlp = LayerNormMLP(latent_dim, out_dim, hidden_dim=hidden_dim, n_layers=3)

    def forward(
        self, latent: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            latent: [*batch, latent_dim]

        Returns:
            Discrete: logits [*batch, action_dim]
            Continuous: (mean [*batch, action_dim], log_std [*batch, action_dim])
                        log_std clamped to [-5, 2]
        """
        out = self.mlp(latent)
        if self.discrete:
            return out
        mean, log_std = out.chunk(2, dim=-1)
        # TODO(plan5): consider softclamp 5*tanh(x/5) to avoid zero gradients at boundary
        log_std = log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std


class Critic(nn.Module):
    """Value network using twohot distribution over bins.

    Mirrors DreamerV3's value head design: categorical cross-entropy over
    symlog-scaled bins, same as RewardHead. Training logic in Plan 5.

    Args:
        latent_dim: Dimension of [h_t; z_t].
        n_bins: Number of twohot bins.
        hidden_dim: MLP width.
    """

    def __init__(
        self,
        latent_dim: int,
        n_bins: int = 255,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.mlp = LayerNormMLP(latent_dim, n_bins, hidden_dim=hidden_dim, n_layers=3)

    def forward(self, latent: Tensor) -> Tensor:
        """Return raw value distribution logits.

        Args:
            latent: [*batch, latent_dim]

        Returns:
            logits: [*batch, n_bins]
        """
        return self.mlp(latent)
