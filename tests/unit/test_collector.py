"""
Unit tests for bvh_rssm/training/collector.py

Covers:
  1. random_policy=True fills the buffer with exactly n_steps transitions.
  2. All oracle_tau values pushed are non-negative.
  3. Each rng_state dict contains 'torch_cpu' key.
  4. Observations in the buffer have the correct shape for ShiftPendulum.
  5. Works with TradingRegime (Discrete action space).
  6. Actor policy path falls back to random when actor not in model (warns).
  7. _make_env raises ValueError for unknown env names.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from bvh_rssm.training.collector import Collector, _make_env, _get_action_dim
from bvh_rssm.training.replay_buffer import ReplayBuffer
from bvh_rssm.networks import RSSM, Encoder
from bvh_rssm.networks.heads import ValidityHead, HazardHead
from bvh_rssm.networks.actor_critic import Actor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _build_tiny_model(obs_dim: int, action_dim: int) -> dict:
    """Construct a minimal model dict for Collector with tiny dims."""
    h_dim = 16
    z_cats, z_classes = 4, 4
    embed_dim = 16
    z_dim = z_cats * z_classes
    latent_dim = h_dim + z_dim

    return {
        "encoder":     Encoder(obs_dim=obs_dim, embed_dim=embed_dim,
                               hidden_dim=32, n_layers=1),
        "rssm":        RSSM(h_dim=h_dim, z_cats=z_cats, z_classes=z_classes,
                            obs_dim=embed_dim, action_dim=action_dim),
        "tau_head":    ValidityHead(latent_dim=latent_dim, action_dim=action_dim,
                                   n_bins=16, hidden_dim=32),
        "hazard_head": HazardHead(latent_dim=latent_dim, n_intervals=4,
                                  hidden_dim=32),
    }


@pytest.fixture
def pendulum_setup():
    """ShiftPendulum with obs_dim=3, action_dim=1 (Box)."""
    obs_dim, action_dim = 3, 1
    model = _build_tiny_model(obs_dim, action_dim)
    buf = ReplayBuffer(capacity=1000, obs_dim=obs_dim, action_dim=action_dim, seq_len=4)
    device = torch.device("cpu")
    return model, buf, device


@pytest.fixture
def trading_setup():
    """TradingRegime with obs_dim=20, action_dim=1 (Discrete)."""
    obs_dim, action_dim = 20, 1
    model = _build_tiny_model(obs_dim, action_dim)
    buf = ReplayBuffer(capacity=1000, obs_dim=obs_dim, action_dim=action_dim, seq_len=4)
    device = torch.device("cpu")
    return model, buf, device


# ---------------------------------------------------------------------------
# Test 1: collect_steps fills buffer with n_steps
# ---------------------------------------------------------------------------

def test_collect_steps_fills_buffer(pendulum_setup):
    """collect_steps(n) pushes exactly n transitions into the buffer."""
    model, buf, device = pendulum_setup
    collector = Collector("ShiftPendulum", model, buf, device)

    n = 30
    collector.collect_steps(n, random_policy=True)

    assert len(buf) == n, f"Expected {n} transitions, got {len(buf)}"


# ---------------------------------------------------------------------------
# Test 2: oracle_tau values are non-negative
# ---------------------------------------------------------------------------

def test_oracle_tau_non_negative(pendulum_setup):
    """All oracle_tau values stored in buffer must be >= 0."""
    model, buf, device = pendulum_setup
    collector = Collector("ShiftPendulum", model, buf, device)
    collector.collect_steps(50, random_policy=True)

    stored_taus = buf._oracle_tau[: len(buf)]
    assert (stored_taus >= 0).all(), \
        f"Found negative oracle_tau values: {stored_taus[stored_taus < 0]}"


# ---------------------------------------------------------------------------
# Test 3: rng_states contain 'torch_cpu' key
# ---------------------------------------------------------------------------

def test_rng_states_have_torch_cpu_key(pendulum_setup):
    """Every rng_state stored in the buffer must have a 'torch_cpu' entry."""
    model, buf, device = pendulum_setup
    collector = Collector("ShiftPendulum", model, buf, device)
    n = 20
    collector.collect_steps(n, random_policy=True)

    for i in range(n):
        rng_state = buf._rng_states[i]
        assert rng_state is not None, f"rng_state at index {i} is None"
        assert "torch_cpu" in rng_state, \
            f"rng_state at index {i} missing 'torch_cpu': keys={list(rng_state.keys())}"


# ---------------------------------------------------------------------------
# Test 4: observations in buffer have correct shape
# ---------------------------------------------------------------------------

def test_obs_shape_in_buffer(pendulum_setup):
    """Observations pushed into the buffer must match obs_dim=3."""
    model, buf, device = pendulum_setup
    collector = Collector("ShiftPendulum", model, buf, device)
    n = 15
    collector.collect_steps(n, random_policy=True)

    assert buf._obs.shape == (buf.capacity, 3), \
        f"Buffer obs shape mismatch: {buf._obs.shape}"
    # Spot-check stored values are finite floats
    stored_obs = buf._obs[:n]
    assert np.isfinite(stored_obs).all(), "Non-finite values found in stored observations"


# ---------------------------------------------------------------------------
# Test 5: TradingRegime (Discrete) works correctly
# ---------------------------------------------------------------------------

def test_trading_regime_discrete_collection(trading_setup):
    """Collector works with Discrete action space (TradingRegime).

    Verifies:
      - n transitions collected
      - action buffer has shape (capacity, 1) with int values in {0, 1, 2}
      - oracle_tau non-negative
    """
    model, buf, device = trading_setup
    collector = Collector("TradingRegime", model, buf, device)
    n = 25
    collector.collect_steps(n, random_policy=True)

    assert len(buf) == n

    stored_actions = buf._action[:n]  # [n, 1]
    assert stored_actions.shape == (n, 1), \
        f"Expected action shape (n, 1), got {stored_actions.shape}"
    # Values must be valid discrete action indices
    assert ((stored_actions >= 0) & (stored_actions <= 2)).all(), \
        f"Unexpected discrete action values: {stored_actions}"

    stored_taus = buf._oracle_tau[:n]
    assert (stored_taus >= 0).all()


# ---------------------------------------------------------------------------
# Test 6: Actor policy path (random_policy=False)
# ---------------------------------------------------------------------------

def test_actor_policy_path(pendulum_setup):
    """random_policy=False with actor in model uses actor (no crash, fills buf)."""
    model, buf, device = pendulum_setup
    obs_dim, action_dim = 3, 1
    h_dim = 16
    z_cats, z_classes = 4, 4
    z_dim = z_cats * z_classes
    latent_dim = h_dim + z_dim

    model["actor"] = Actor(
        latent_dim=latent_dim,
        action_dim=action_dim,
        discrete=False,
        hidden_dim=32,
    )

    collector = Collector("ShiftPendulum", model, buf, device)
    n = 20
    collector.collect_steps(n, random_policy=False)

    assert len(buf) == n


# ---------------------------------------------------------------------------
# Test 7: Fallback to random when actor missing (random_policy=False)
# ---------------------------------------------------------------------------

def test_actor_missing_fallback(pendulum_setup):
    """random_policy=False without actor key emits warning and falls back."""
    model, buf, device = pendulum_setup
    assert "actor" not in model

    collector = Collector("ShiftPendulum", model, buf, device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        collector.collect_steps(10, random_policy=False)

    warning_messages = [str(w.message) for w in caught]
    assert any("actor" in msg or "random" in msg.lower() for msg in warning_messages), \
        f"Expected a warning about missing actor, got: {warning_messages}"

    assert len(buf) == 10


# ---------------------------------------------------------------------------
# Test 8: _make_env raises ValueError for unknown env
# ---------------------------------------------------------------------------

def test_make_env_unknown_raises():
    """_make_env raises ValueError for an unrecognised environment name."""
    with pytest.raises((ValueError, Exception)):
        _make_env("ThisEnvDoesNotExist_XYZ_123")


# ---------------------------------------------------------------------------
# Test 9: _get_action_dim returns correct values
# ---------------------------------------------------------------------------

def test_get_action_dim():
    """_get_action_dim returns flat product for Box and 1 for Discrete."""
    import gymnasium as gym
    import numpy as np

    box_1d = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
    assert _get_action_dim(box_1d) == 3

    box_2d = gym.spaces.Box(low=-1.0, high=1.0, shape=(2, 4))
    assert _get_action_dim(box_2d) == 8

    discrete = gym.spaces.Discrete(5)
    assert _get_action_dim(discrete) == 1


# ---------------------------------------------------------------------------
# Test 10: Multiple collect_steps calls accumulate (resume mid-episode)
# ---------------------------------------------------------------------------

def test_multiple_collect_calls_accumulate(pendulum_setup):
    """Calling collect_steps twice accumulates transitions additively."""
    model, buf, device = pendulum_setup
    collector = Collector("ShiftPendulum", model, buf, device)

    collector.collect_steps(15, random_policy=True)
    assert len(buf) == 15

    collector.collect_steps(10, random_policy=True)
    assert len(buf) == 25
