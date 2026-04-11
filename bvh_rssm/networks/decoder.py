"""
Decoder — observation reconstruction p(o_t | h_t, z_t).

Takes concatenated latent [h_t; z_t] and predicts observation distribution
parameters. Mean is in symlog space; symexp applied to recover original scale.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from bvh_rssm.networks.common import LayerNormMLP
from bvh_rssm.utils import symexp


class Decoder(nn.Module):
    """Observation decoder: latent [h; z] → obs_mean (original scale).

    Predicts a Gaussian observation distribution. Mean in symlog space,
    symexp applied to recover original scale. Log-std is a single scalar
    (isotropic Gaussian) expanded to obs_dim for convenience.

    Args:
        latent_dim: Dimension of concatenated [h_t; z_t].
        obs_dim: Target observation dimensionality.
        hidden_dim: MLP hidden layer width.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.mlp = LayerNormMLP(
            in_dim=latent_dim,
            out_dim=obs_dim + 1,  # obs_dim means + 1 shared log_std scalar
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        self.obs_dim = obs_dim

    def forward(self, latent: Tensor) -> Tensor:
        """Decode latent to observation mean in original scale (for rendering).

        Returns only the mean with symexp applied to recover original scale.
        Callers who need distribution parameters (mean_symlog, log_std) for
        training loss computation should use decode_symlog() instead.

        Args:
            latent: Concatenated [h_t; z_t] of shape [*batch, latent_dim].

        Returns:
            mean of shape [*batch, obs_dim] in original observation scale
            (symexp applied). Do NOT pair this with log_std from decode_symlog()
            to construct a Normal — the two are in different scales.
        """
        out = self.mlp(latent)                            # [*batch, obs_dim+1]
        mean_symlog = out[..., :self.obs_dim]             # [*batch, obs_dim]
        mean = symexp(mean_symlog)                        # recover original scale
        return mean

    def decode_symlog(self, latent: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode latent to symlog-space distribution parameters (for loss computation).

        This is the method to use for training loss — compute NLL against
        symlog-transformed targets: loss = NLL(symlog(obs_target), mean_symlog, std).

        DreamerV3 computes all reconstruction losses in symlog space for numerical
        stability. Use forward() only for rendering/visualization where original
        scale is needed.

        Args:
            latent: Concatenated [h_t; z_t] of shape [*batch, latent_dim].

        Returns:
            (mean_symlog, log_std) each of shape [*batch, obs_dim].
            mean_symlog is in symlog space (no symexp applied).
            log_std is clamped to [-5, 5].
        """
        out = self.mlp(latent)
        mean_symlog = out[..., :self.obs_dim]
        log_std_scalar = out[..., self.obs_dim:]
        log_std = log_std_scalar.expand(
            *out.shape[:-1], self.obs_dim
        ).clamp(-5.0, 5.0)
        return mean_symlog, log_std
