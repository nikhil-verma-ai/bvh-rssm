"""
Standard DreamerV3 heads: RewardHead and ContinueHead.

Both heads take the concatenated latent [h_t; z_t] as input.
Novel BVH heads (ValidityHead, HazardHead) are added in Plan 4.

RewardHead:
  - Output: categorical distribution over n_bins (twohot encoding)
  - Loss: categorical cross-entropy between predicted logits and twohot(symlog(r_t))
  - Decode: expected value via dot product with bin centers → symexp → original scale

ContinueHead:
  - Output: sigmoid probability of episode continuation
  - Loss: binary cross-entropy with continue flag c_t ∈ {0, 1}
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bvh_rssm.networks.common import MLP
from bvh_rssm.utils import symlog, symexp, twohot, twohot_decode


class RewardHead(nn.Module):
    """Reward prediction head using twohot encoding.

    Predicts a distribution over reward magnitude via categorical cross-entropy.
    This decouples gradient magnitude from reward scale — DreamerV3's key
    robustness trick for multi-domain training.

    Args:
        latent_dim: Dimension of concatenated [h_t; z_t].
        n_bins: Number of twohot bins (DreamerV3 default: 255).
        hidden_dim: MLP hidden width.
    """

    def __init__(
        self,
        latent_dim: int,
        n_bins: int = 255,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.mlp = MLP(latent_dim, n_bins, hidden_dim=hidden_dim, n_layers=2)

    def forward(self, latent: Tensor) -> Tensor:
        """Return raw logits over reward bins.

        Args:
            latent: [*batch, latent_dim]

        Returns:
            logits: [*batch, n_bins]
        """
        return self.mlp(latent)

    def decode(self, logits: Tensor, bins: Tensor) -> Tensor:
        """Decode logits to expected reward in original scale.

        Args:
            logits: [*batch, n_bins]
            bins: [n_bins] bin centers in symlog space

        Returns:
            reward: [*batch] in original scale (symexp applied)
        """
        probs = logits.softmax(-1)
        reward_symlog = twohot_decode(probs, bins)
        return symexp(reward_symlog)

    def loss(self, latent: Tensor, target_reward: Tensor, bins: Tensor) -> Tensor:
        """Compute twohot categorical cross-entropy loss.

        Args:
            latent: [B, latent_dim]
            target_reward: [B] raw reward values
            bins: [n_bins] symlog-space bin centers

        Returns:
            Scalar loss.
        """
        logits = self.forward(latent)                         # [B, n_bins]
        target_symlog = symlog(target_reward)                 # [B]
        target_twohot = twohot(target_symlog, bins)           # [B, n_bins]
        log_probs = F.log_softmax(logits, dim=-1)             # [B, n_bins]
        return -(target_twohot * log_probs).sum(-1).mean()


class ContinueHead(nn.Module):
    """Episode continuation prediction head.

    Predicts P(episode continues) at each step. Binary cross-entropy loss
    with target c_t = 1 if episode continues, 0 if terminated.

    Args:
        latent_dim: Dimension of concatenated [h_t; z_t].
        hidden_dim: MLP hidden width.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.mlp = MLP(latent_dim, 1, hidden_dim=hidden_dim, n_layers=2)

    def forward(self, latent: Tensor) -> Tensor:
        """Return raw logits for continuation probability.

        Args:
            latent: [*batch, latent_dim]

        Returns:
            logits: [*batch, 1]
        """
        return self.mlp(latent)

    def probability(self, latent: Tensor) -> Tensor:
        """Return continuation probability (sigmoid of logits).

        Args:
            latent: [*batch, latent_dim]

        Returns:
            prob: [*batch, 1] in [0, 1]
        """
        return torch.sigmoid(self.forward(latent))

    def loss(self, latent: Tensor, continue_flag: Tensor) -> Tensor:
        """Binary cross-entropy loss.

        Args:
            latent: [B, latent_dim]
            continue_flag: [B, 1] with values 0.0 or 1.0

        Returns:
            Scalar loss.
        """
        logits = self.forward(latent)
        return F.binary_cross_entropy_with_logits(logits, continue_flag)
