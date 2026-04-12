"""
AdaptivePolicyRouter — classifies validity state and derives imagination horizon.

Uses the survival curve S(t) from HazardHead to derive two thresholds:
  tau_hi:  first interval index where S(t) <= 0.80 (80th-pct survival time)
  tau_min: first interval index where S(t) <= 0.20 (20th-pct survival time)

Classification of tau_hat (scalar float, expected validity steps):
  HIGH  : tau_hat > tau_hi  — confident, use full imagination horizon
  DIM   : tau_min <= tau_hat <= tau_hi  — uncertain, halve the horizon
  STALE : tau_hat < tau_min  — expired, trigger refresh (horizon=1)

All methods are stateless. No nn.Module. No stored tensors.
"""
from __future__ import annotations

import enum
import torch
from torch import Tensor


class RouterState(enum.Enum):
    HIGH = "HIGH"
    DIM = "DIM"
    STALE = "STALE"


class AdaptivePolicyRouter:
    """Stateless router that maps validity estimates to policy decisions.

    No constructor arguments — thresholds are derived fresh from each survival
    curve, so the router works correctly even as the hazard head improves during
    training.
    """

    def thresholds_from_survival(self, S: Tensor) -> tuple[int, int]:
        """Derive (tau_hi, tau_min) from a 1-D survival curve.

        Searches left-to-right for the first index where S(t) crosses each
        threshold. If the threshold is never crossed, returns K-1 as a safe
        fallback (the model is very confident; keep the last valid index).

        Args:
            S: Non-increasing survival probabilities of shape [K].

        Returns:
            (tau_hi, tau_min): Integer indices (0-based) where:
              tau_hi  = first t where S[t] <= 0.80
              tau_min = first t where S[t] <= 0.20
        """
        K = S.shape[0]

        # torch.nonzero returns a [N, 1] tensor of indices; squeeze to 1-D.
        # The mask S <= threshold is True at all crossing points; we want the
        # first (leftmost) such index.
        hi_mask = (S <= 0.80).nonzero(as_tuple=False).squeeze(-1)
        tau_hi: int = int(hi_mask[0].item()) if hi_mask.numel() > 0 else K - 1

        min_mask = (S <= 0.20).nonzero(as_tuple=False).squeeze(-1)
        tau_min: int = int(min_mask[0].item()) if min_mask.numel() > 0 else K - 1

        return tau_hi, tau_min

    def classify(self, tau_hat: float, S: Tensor) -> RouterState:
        """Classify current validity state.

        tau_hi is the first index where S(t) <= 0.80 (optimistic crossing, lower index).
        tau_min is the first index where S(t) <= 0.20 (pessimistic crossing, higher index).
        In index-space: tau_hi <= tau_min always.

        Classification (comparing tau_hat against survival index thresholds):
          STALE : tau_hat < tau_min  — below the pessimistic threshold, trigger refresh
          DIM   : tau_hi <= tau_hat <= tau_min  — uncertain, halve the horizon
          HIGH  : tau_hat > tau_min  — confident, use full imagination horizon

        Args:
            tau_hat: Expected validity horizon (scalar float, step-space).
            S: 1-D survival curve [K] from HazardHead.survival(latent)[b].

        Returns:
            RouterState: HIGH, DIM, or STALE.
        """
        tau_hi, tau_min = self.thresholds_from_survival(S)
        # Check STALE first: tau_hat strictly below the pessimistic (tau_min) boundary.
        # Note: tau_hi <= tau_min in index-space, so we must check tau_min before tau_hi
        # to avoid misclassifying the DIM region [tau_hi, tau_min) as HIGH.
        if tau_hat > tau_min:
            return RouterState.HIGH
        elif tau_hat < tau_hi:
            return RouterState.STALE
        else:
            # tau_hi <= tau_hat <= tau_min (inclusive on both ends)
            return RouterState.DIM

    def imagination_horizon(
        self,
        state: RouterState,
        tau_hat: float,
        full_horizon: int = 16,
    ) -> int:
        """Return the number of imagination steps to use.

        Args:
            state: Classification from classify().
            tau_hat: Expected validity horizon (scalar float).
            full_horizon: Maximum horizon used when state is HIGH.

        Returns:
            Number of imagination steps:
              HIGH  -> full_horizon
              DIM   -> max(1, int(tau_hat / 2))
              STALE -> 1
        """
        if state == RouterState.HIGH:
            return full_horizon
        elif state == RouterState.STALE:
            return 1
        else:
            # DIM: halve the horizon, minimum 1
            return max(1, int(tau_hat / 2))
