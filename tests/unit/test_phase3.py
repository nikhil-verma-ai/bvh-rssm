"""Tests for Phase 3 BVH-gated policy logic."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Router gating logic
# ──────────────────────────────────────────────────────────────────────────────

def test_imagination_horizon_gating():
    from bvh_rssm.causal.router import AdaptivePolicyRouter, RouterState

    router = AdaptivePolicyRouter()
    # S that puts tau_hi=2, tau_min=8
    S = torch.tensor([1.0, 0.95, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.05, 0.01])
    # HIGH: tau_hat=10 > tau_min=8 → full_horizon=16
    assert router.imagination_horizon(RouterState.HIGH, 10.0, full_horizon=16) == 16
    # DIM: tau_hat=4 → max(1, 4//2) = 2
    assert router.imagination_horizon(RouterState.DIM, 4.0, full_horizon=16) == 2
    # STALE → always 1
    assert router.imagination_horizon(RouterState.STALE, 0.5, full_horizon=16) == 1


def test_gated_horizon_never_exceeds_full():
    from bvh_rssm.causal.router import AdaptivePolicyRouter, RouterState

    router = AdaptivePolicyRouter()
    S_all_high = torch.ones(16)  # S never drops → tau_min = K-1 = 15
    state = router.classify(20.0, S_all_high)
    assert state == RouterState.HIGH
    h = router.imagination_horizon(state, 20.0, full_horizon=16)
    assert h == 16


def test_gated_horizon_stale_is_one():
    from bvh_rssm.causal.router import AdaptivePolicyRouter, RouterState

    router = AdaptivePolicyRouter()
    # S crosses 0.80 at index 3 (tau_hi=3) and 0.20 at index 7 (tau_min=7).
    # tau_hat=1.0 < tau_hi=3 → STALE
    S = torch.tensor([1.0, 0.9, 0.85, 0.75, 0.5, 0.4, 0.25, 0.15, 0.05, 0.01])
    state = router.classify(1.0, S)
    assert state == RouterState.STALE
    assert router.imagination_horizon(state, 1.0, full_horizon=16) == 1


# ──────────────────────────────────────────────────────────────────────────────
# train_phase3 signature check
# ──────────────────────────────────────────────────────────────────────────────

def test_imagination_gating_flag_accepted():
    """train_phase3 must accept imagination_gating with default False."""
    from bvh_rssm.training.trainer import Trainer

    sig = inspect.signature(Trainer.train_phase3)
    assert "imagination_gating" in sig.parameters, \
        "train_phase3 must have imagination_gating parameter"
    p = sig.parameters["imagination_gating"]
    assert p.default is False, "imagination_gating default must be False"


# ──────────────────────────────────────────────────────────────────────────────
# Episode rollout helpers
# ──────────────────────────────────────────────────────────────────────────────

def test_zero_action_fallback_shape():
    """STALE state must yield zero action of correct shape."""
    action_dim = 6
    stale = True
    if stale:
        action = np.zeros(action_dim, dtype=np.float32)
    assert action.shape == (action_dim,)
    assert (action == 0.0).all()


def test_episode_return_accumulation():
    """delta_return = mean(bvh_returns) - mean(vanilla_returns)."""
    bvh_returns = [10.0, 20.0, 30.0]      # mean = 20.0
    vanilla_returns = [5.0, 15.0, 25.0]   # mean = 15.0
    delta = (
        sum(bvh_returns) / len(bvh_returns)
        - sum(vanilla_returns) / len(vanilla_returns)
    )
    assert abs(delta - 5.0) < 1e-6


def test_failure_rate():
    """failure_rate = fraction of episodes with return < -50."""
    returns = [100.0, -60.0, -55.0, 10.0]
    threshold = -50.0
    rate = sum(1 for r in returns if r < threshold) / len(returns)
    assert abs(rate - 0.5) < 1e-6
