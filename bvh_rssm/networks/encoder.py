"""
Encoder — observation embedding for RSSM posterior q(z_t | h_t, o_t).

Applies symlog to inputs (per DreamerV3) to handle observations with
varying scales without per-environment normalization.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from bvh_rssm.networks.common import LayerNormMLP
from bvh_rssm.utils import symlog


class Encoder(nn.Module):
    """Observation encoder: raw obs → latent embedding.

    Applies symlog to compress observation scale, then passes through a
    LayerNormMLP. Output fed into RSSM.observe() as obs_embed.

    Contract: Expects raw, unscaled observations. The symlog transform is applied
    internally to compress observation scale. Do NOT pre-normalize observations
    before passing to this encoder — symlog handles scale compression.
    Environments that already return compressed representations (log-returns,
    normalized states) will have symlog applied redundantly, but symlog(x) ≈ x
    for |x| << 1 so the numerical impact is negligible.

    Args:
        obs_dim: Raw observation dimensionality.
        embed_dim: Output embedding dimension (= RSSM obs_dim).
        hidden_dim: MLP hidden layer width.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.mlp = LayerNormMLP(
            in_dim=obs_dim,
            out_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    def forward(self, obs: Tensor) -> Tensor:
        """Encode raw observations to latent embedding.

        Args:
            obs: Raw observations of shape [*batch, obs_dim].

        Returns:
            Embedding of shape [*batch, embed_dim].
        """
        return self.mlp(symlog(obs))
