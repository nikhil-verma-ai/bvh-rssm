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
from bvh_rssm.utils import symlog, symexp, symlog_bins, twohot, twohot_decode


class RewardHead(nn.Module):
    """Reward prediction head using twohot encoding.

    Predicts a distribution over reward magnitude via categorical cross-entropy.
    This decouples gradient magnitude from reward scale — DreamerV3's key
    robustness trick for multi-domain training.

    Bin centers are stored as a registered buffer so they automatically follow
    the module to the correct device via `.to(device)` / `.cuda()`.

    Args:
        latent_dim: Dimension of concatenated [h_t; z_t].
        n_bins: Number of twohot bins (DreamerV3 default: 255).
        hidden_dim: MLP hidden width.
        bins: Optional pre-constructed [n_bins] bin-center tensor. When None
            (default) bins are created via ``symlog_bins(n_bins)``.
    """

    def __init__(
        self,
        latent_dim: int,
        n_bins: int = 255,
        hidden_dim: int = 512,
        bins: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.mlp = MLP(latent_dim, n_bins, hidden_dim=hidden_dim, n_layers=2)
        self.register_buffer(
            'bins',
            bins if bins is not None else symlog_bins(n_bins),
        )

    def forward(self, latent: Tensor) -> Tensor:
        """Return raw logits over reward bins.

        Args:
            latent: [*batch, latent_dim]

        Returns:
            logits: [*batch, n_bins]
        """
        return self.mlp(latent)

    def decode(self, logits: Tensor) -> Tensor:
        """Decode logits to expected reward in original scale.

        Uses ``self.bins`` (registered buffer) for bin centers; no explicit
        device management required by the caller.

        Args:
            logits: [*batch, n_bins]

        Returns:
            reward: [*batch] in original scale (symexp applied)
        """
        probs = logits.softmax(-1)
        reward_symlog = twohot_decode(probs, self.bins)
        return symexp(reward_symlog)

    def loss(self, latent: Tensor, target_reward: Tensor) -> Tensor:
        """Compute twohot categorical cross-entropy loss.

        Uses ``self.bins`` (registered buffer) for bin centers; no explicit
        device management required by the caller.

        Args:
            latent: [B, latent_dim]
            target_reward: [B] raw reward values

        Returns:
            Scalar loss.
        """
        logits = self.forward(latent)                         # [B, n_bins]
        target_symlog = symlog(target_reward)                 # [B]
        target_twohot = twohot(target_symlog, self.bins)      # [B, n_bins]
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

        The trailing singleton dimension produced by the linear output is
        squeezed so the returned tensor shape matches [*batch], consistent
        with ``RewardHead.decode``.

        Args:
            latent: [*batch, latent_dim]

        Returns:
            prob: [*batch] in [0, 1]
        """
        return torch.sigmoid(self.forward(latent)).squeeze(-1)

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


class ValidityHead(nn.Module):
    """Validity horizon head (τ-head).

    Predicts the number of steps until the world model's imagined latent
    diverges from the posterior by more than ε nats (KL threshold).

    Input: concatenated latent [h_t; z_t] + action embedding.
    Output: categorical distribution over n_bins via twohot encoding.

    stop_grad controls whether gradients flow back into the RSSM latent:
    - Phase 2: stop_grad=True (heads train on frozen world model)
    - Phase 3: stop_grad=False (joint fine-tuning)

    Args:
        latent_dim: Dimension of [h_t; z_t].
        action_dim: Action space dimension.
        n_bins: Number of twohot bins for horizon distribution (default 255).
        embed_dim: Action embedding dimension.
        hidden_dim: MLP hidden width.
        max_horizon: Maximum predictable horizon in steps (default 1000).
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        n_bins: int = 255,
        embed_dim: int = 64,
        hidden_dim: int = 512,
        max_horizon: int = 1000,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.action_embed = MLP(action_dim, embed_dim, hidden_dim=hidden_dim, n_layers=1)
        self.mlp = MLP(latent_dim + embed_dim, n_bins, hidden_dim=hidden_dim, n_layers=2)
        # Bins in symlog space over [0, symlog(max_horizon)]
        symlog_hi = symlog(torch.tensor(float(max_horizon))).item()
        self.register_buffer(
            "bins",
            symlog_bins(n_bins, lo=0.0, hi=float(symlog_hi))
        )

    def forward(self, latent: Tensor, action: Tensor, stop_grad: bool = False) -> Tensor:
        """Return raw logits over horizon bins.

        Args:
            latent: [*batch, latent_dim]
            action: [*batch, action_dim]
            stop_grad: If True, detach latent before processing (Phase 2).

        Returns:
            logits: [*batch, n_bins]
        """
        if stop_grad:
            latent = latent.detach()
        a_embed = self.action_embed(action)
        x = torch.cat([latent, a_embed], dim=-1)
        return self.mlp(x)

    def decode(self, logits: Tensor) -> Tensor:
        """Decode logits to expected τ̂ in steps (original scale, >= 0).

        Args:
            logits: [*batch, n_bins]

        Returns:
            tau_hat: [*batch]
        """
        probs = logits.softmax(-1)
        tau_symlog = twohot_decode(probs, self.bins)
        return symexp(tau_symlog).clamp(min=0.0)

    def loss(
        self,
        latent: Tensor,
        action: Tensor,
        oracle_tau: Tensor,
        stop_grad: bool = False,
    ) -> Tensor:
        """Twohot cross-entropy loss against oracle τ*.

        Args:
            latent: [B, latent_dim]
            action: [B, action_dim]
            oracle_tau: [B] ground-truth steps-to-shift
            stop_grad: If True, detach latent (Phase 2 training mode).

        Returns:
            Scalar loss.
        """
        logits = self.forward(latent, action, stop_grad=stop_grad)
        target_symlog = symlog(oracle_tau.float())
        target_twohot = twohot(target_symlog, self.bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_twohot * log_probs).sum(-1).mean()


class HazardHead(nn.Module):
    """Hazard head (λ-head) — discrete-time survival with competing risks.

    Models the survival function S(t) = P(shift happens after t steps).
    Three competing-risk sub-heads (Sources A, B, C) each output K independent
    hazard probabilities h_i ∈ (0, 1).

    Combined hazard: h_total(i) = 1 - ∏_{X∈{A,B,C}} (1 - h_X(i))
    Survival function: S(t) = ∏_{i=0}^{t-1} (1 - h_total(i))

    Phase 1/2: Source B only — Sources A and C are zero-initialized stubs.
    Full pycox integration for survival likelihood is in training/losses.py.

    Args:
        latent_dim: Dimension of [h_t; z_t].
        n_intervals: Number of discrete time intervals (K=16 per spec).
        hidden_dim: MLP hidden width.
    """

    def __init__(
        self,
        latent_dim: int,
        n_intervals: int = 16,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.n_intervals = n_intervals
        self.source_a = MLP(latent_dim, n_intervals, hidden_dim=hidden_dim, n_layers=2)
        self.source_b = MLP(latent_dim, n_intervals, hidden_dim=hidden_dim, n_layers=2)
        self.source_c = MLP(latent_dim, n_intervals, hidden_dim=hidden_dim, n_layers=2)
        # Initialize sources A and C to near-zero output (stubs for Phase 1/2).
        # sigmoid(-5) ≈ 0.007 — effectively zero contribution to combined hazard.
        with torch.no_grad():
            for m in [self.source_a, self.source_c]:
                # Only zero the final projection layer — keep intermediate Kaiming init
                # so sources A/C have a viable weight landscape when activated in Phase 3
                final_layer = m.net[-1]
                assert isinstance(final_layer, nn.Linear), \
                    f"Expected final MLP layer to be Linear, got {type(final_layer)}"
                final_layer.weight.data.zero_()
                final_layer.bias.data.fill_(-5.0)  # sigmoid(-5) ≈ 0.007

    def forward(self, latent: Tensor):
        """Return per-source hazard probabilities.

        Args:
            latent: [*batch, latent_dim]

        Returns:
            (h_A, h_B, h_C): each [*batch, n_intervals] in (0, 1)
        """
        h_A = torch.sigmoid(self.source_a(latent))
        h_B = torch.sigmoid(self.source_b(latent))
        h_C = torch.sigmoid(self.source_c(latent))
        return h_A, h_B, h_C

    def combined_hazard(self, latent: Tensor) -> Tensor:
        """h_total(i) = 1 - ∏_{X}(1 - h_X(i)).

        Args:
            latent: [*batch, latent_dim]

        Returns:
            h_total: [*batch, n_intervals] in (0, 1)
        """
        h_A, h_B, h_C = self.forward(latent)
        return 1.0 - (1.0 - h_A) * (1.0 - h_B) * (1.0 - h_C)

    def survival(self, latent: Tensor) -> Tensor:
        """S(t) = ∏_{i<t}(1 - h_total(i)) — monotonically non-increasing.

        Args:
            latent: [*batch, latent_dim]

        Returns:
            S: [*batch, n_intervals]
        """
        h_total = self.combined_hazard(latent)
        return torch.cumprod(1.0 - h_total, dim=-1)

    def loss_source_b(
        self,
        latent: Tensor,
        event_times: Tensor,
        event_occurred: Tensor,
    ) -> Tensor:
        """BCE-approximated survival loss for Source B (Phase 1/2).

        Builds binary targets: d[b, i] = 1 iff shift occurred at interval i.
        Censored samples only contribute up to their censoring interval.

        Args:
            latent: [B, latent_dim]
            event_times: [B] interval index (0-indexed)
            event_occurred: [B] bool, True if observed, False if censored

        Returns:
            Scalar loss.
        """
        _, h_B, _ = self.forward(latent)
        B, K = h_B.shape

        t = event_times.long().clamp(0, K - 1)                          # [B]
        arange = torch.arange(K, device=latent.device).unsqueeze(0)     # [1, K]

        # mask[b, i] = 1 iff i <= t[b]
        mask = (arange <= t.unsqueeze(1)).float()                        # [B, K]

        # targets[b, t[b]] = 1 iff event_occurred[b]
        targets = torch.zeros(B, K, device=latent.device)
        obs_idx = event_occurred.bool().nonzero(as_tuple=True)[0]        # indices of observed events
        if obs_idx.numel() > 0:
            targets[obs_idx, t[obs_idx]] = 1.0

        bce = F.binary_cross_entropy(h_B, targets, reduction="none")
        return (bce * mask).sum() / mask.sum().clamp(min=1.0)
