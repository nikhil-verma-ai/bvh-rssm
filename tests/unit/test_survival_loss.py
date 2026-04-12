"""
Unit tests for the proper discrete-time survival NLL loss and Phase 3 trainer.

Tests verify:
  1. survival_loss with all events observed at t=0 equals -log(h(0)).
  2. survival_loss with all censored equals cumulative log-survival only.
  3. survival_loss >= 0 (NLL is non-negative).
  4. HazardHead.loss() runs without error.
  5. train_phase3() runs one iteration on a tiny model without error.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from bvh_rssm.networks.heads import HazardHead
from bvh_rssm.training.losses import survival_loss


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

LATENT_DIM = 32
N_INTERVALS = 8
B = 4


def make_hazard_head(latent_dim: int = LATENT_DIM, n_intervals: int = N_INTERVALS) -> HazardHead:
    return HazardHead(latent_dim=latent_dim, n_intervals=n_intervals, hidden_dim=64)


def make_latent(B: int = B, latent_dim: int = LATENT_DIM) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, latent_dim)


# ---------------------------------------------------------------------------
# Task 1 tests: survival_loss correctness
# ---------------------------------------------------------------------------


class TestSurvivalLossAllObservedAtZero:
    """All events observed at interval 0 → loss == -mean(log h(0))."""

    def test_all_observed_at_t0_source_b(self):
        """
        When every sample has event_time=0 and event_occurred=True:
          log L_b = log h_B(b, 0) + cs[b, 0]
        cs[b, 0] = 0 by construction (no intervals before t=0).
        So loss = -mean(log h_B[:, 0]).

        We construct a head whose source-B output at interval 0 is exactly p,
        then verify the loss matches -log(p) for a known p.
        """
        head = make_hazard_head()
        latent = make_latent()

        event_times = torch.zeros(B, dtype=torch.long)      # all at interval 0
        event_occurred = torch.ones(B, dtype=torch.bool)    # all observed

        loss = survival_loss(head, latent, event_times, event_occurred, use_all_sources=False)

        # Manually compute expected value using the same head's source_B output
        with torch.no_grad():
            _, h_B, _ = head.forward(latent)
            h_B_clamped = h_B.clamp(1e-7, 1.0 - 1e-7)
            # At t=0: cs[b, 0] = 0, so log L_b = log h_B[b, 0]
            expected = -torch.log(h_B_clamped[:, 0]).mean()

        assert torch.isfinite(loss), "Loss must be finite"
        assert torch.allclose(loss, expected, atol=1e-5), (
            f"Expected loss={expected.item():.6f}, got {loss.item():.6f}"
        )

    def test_all_observed_at_t0_combined(self):
        """Same check but with use_all_sources=True (combined hazard)."""
        head = make_hazard_head()
        latent = make_latent()

        event_times    = torch.zeros(B, dtype=torch.long)
        event_occurred = torch.ones(B, dtype=torch.bool)

        loss = survival_loss(head, latent, event_times, event_occurred, use_all_sources=True)

        with torch.no_grad():
            h_total = head.combined_hazard(latent).clamp(1e-7, 1.0 - 1e-7)
            expected = -torch.log(h_total[:, 0]).mean()

        assert torch.isfinite(loss)
        assert torch.allclose(loss, expected, atol=1e-5)


class TestSurvivalLossAllCensored:
    """All samples censored → loss = -mean(sum_{i=0}^{t_b} log(1 - h(i)))."""

    def test_all_censored_t1(self):
        """
        Censored at t=1: log L_b = cs[b, 2] = log(1-h[b,0]) + log(1-h[b,1]).
        """
        head = make_hazard_head()
        latent = make_latent()

        t_censor = 1
        event_times    = torch.full((B,), t_censor, dtype=torch.long)
        event_occurred = torch.zeros(B, dtype=torch.bool)  # all censored

        loss = survival_loss(head, latent, event_times, event_occurred, use_all_sources=False)

        with torch.no_grad():
            _, h_B, _ = head.forward(latent)
            h_B_c = h_B.clamp(1e-7, 1.0 - 1e-7)
            log_surv = torch.log(1.0 - h_B_c)              # [B, K]
            # Censored at t=1 → sum i=0..1 = log_surv[:,0] + log_surv[:,1]
            expected = -log_surv[:, :t_censor + 1].sum(-1).mean()

        assert torch.isfinite(loss)
        assert torch.allclose(loss, expected, atol=1e-5), (
            f"Expected loss={expected.item():.6f}, got {loss.item():.6f}"
        )

    def test_all_censored_at_last_interval(self):
        """Censored at K-1 (max interval): only survival contribution, no hazard term."""
        head = make_hazard_head()
        latent = make_latent()

        event_times    = torch.full((B,), N_INTERVALS - 1, dtype=torch.long)
        event_occurred = torch.zeros(B, dtype=torch.bool)

        loss = survival_loss(head, latent, event_times, event_occurred, use_all_sources=False)

        assert torch.isfinite(loss)
        assert loss.item() >= 0.0


class TestSurvivalLossNonNegative:
    """NLL must be >= 0 for all valid inputs."""

    @pytest.mark.parametrize("use_all", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 42, 123])
    def test_nll_nonneg(self, use_all, seed):
        torch.manual_seed(seed)
        head = make_hazard_head()
        latent = torch.randn(B, LATENT_DIM)
        event_times    = torch.randint(0, N_INTERVALS, (B,))
        event_occurred = torch.randint(0, 2, (B,)).bool()

        loss = survival_loss(head, latent, event_times, event_occurred, use_all_sources=use_all)
        assert loss.item() >= 0.0, f"NLL must be >= 0, got {loss.item()}"

    def test_nll_is_finite_with_extreme_hazards(self):
        """Clamping must prevent NaN/inf even when network output is extreme."""
        # Construct a head whose source_b saturates near 0 or 1
        head = make_hazard_head()
        with torch.no_grad():
            # Force source_b weights to large positive values → sigmoid ≈ 1
            for p in head.source_b.parameters():
                p.data.fill_(10.0)

        latent = make_latent()
        event_times    = torch.randint(0, N_INTERVALS, (B,))
        event_occurred = torch.randint(0, 2, (B,)).bool()

        loss = survival_loss(head, latent, event_times, event_occurred, use_all_sources=False)
        assert torch.isfinite(loss), f"Loss must be finite with saturated hazards, got {loss}"


class TestSurvivalLossDifferentiable:
    """Gradients must flow through survival_loss."""

    @pytest.mark.parametrize("use_all", [False, True])
    def test_gradients_flow(self, use_all):
        head = make_hazard_head()
        latent = make_latent().requires_grad_(True)
        event_times    = torch.randint(0, N_INTERVALS, (B,))
        event_occurred = torch.randint(0, 2, (B,)).bool()

        loss = survival_loss(head, latent, event_times, event_occurred, use_all_sources=use_all)
        loss.backward()

        assert latent.grad is not None, "Gradient must exist on latent"
        assert latent.grad.abs().sum() > 0, "Gradient must be non-zero"


# ---------------------------------------------------------------------------
# Task 2 tests: HazardHead.loss() wrapper
# ---------------------------------------------------------------------------


class TestHazardHeadLoss:
    """HazardHead.loss() thin-wrapper tests."""

    def test_loss_runs_without_error(self):
        head = make_hazard_head()
        latent = make_latent()
        event_times    = torch.randint(0, N_INTERVALS, (B,))
        event_occurred = torch.randint(0, 2, (B,)).bool()

        loss = head.loss(latent, event_times, event_occurred, use_all_sources=False)
        assert torch.isfinite(loss)

    def test_loss_matches_survival_loss_function(self):
        """HazardHead.loss() must return the same value as survival_loss()."""
        head = make_hazard_head()
        latent = make_latent()
        event_times    = torch.tensor([0, 1, 3, 7])
        event_occurred = torch.tensor([True, False, True, False])

        for use_all in [False, True]:
            head_loss = head.loss(latent, event_times, event_occurred, use_all_sources=use_all)
            fn_loss   = survival_loss(head, latent, event_times, event_occurred, use_all_sources=use_all)
            assert torch.allclose(head_loss, fn_loss), (
                f"HazardHead.loss and survival_loss disagree for use_all={use_all}"
            )

    def test_loss_differentiable(self):
        head = make_hazard_head()
        latent = make_latent().requires_grad_(True)
        event_times    = torch.randint(0, N_INTERVALS, (B,))
        event_occurred = torch.randint(0, 2, (B,)).bool()

        loss = head.loss(latent, event_times, event_occurred, use_all_sources=True)
        loss.backward()

        assert latent.grad is not None
        assert latent.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Task 3 tests: train_phase3() smoke test
# ---------------------------------------------------------------------------


def _make_tiny_model(obs_dim: int = 3, action_dim: int = 1):
    """Build the smallest possible valid BVH-RSSM model for smoke testing."""
    from bvh_rssm.networks import RSSM, Encoder, Decoder, RewardHead, ContinueHead
    from bvh_rssm.networks.heads import ValidityHead, HazardHead
    from bvh_rssm.networks.actor_critic import Actor, Critic

    h_dim      = 16
    z_cats     = 4
    z_classes  = 4
    embed_dim  = 32
    hidden_dim = 32
    n_bins     = 16
    n_intervals = 4

    z_dim      = z_cats * z_classes
    latent_dim = h_dim + z_dim

    return {
        "encoder":      Encoder(obs_dim=obs_dim, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=1),
        "decoder":      Decoder(latent_dim=latent_dim, obs_dim=obs_dim, hidden_dim=hidden_dim, n_layers=1),
        "rssm":         RSSM(h_dim=h_dim, z_cats=z_cats, z_classes=z_classes,
                             obs_dim=embed_dim, action_dim=action_dim),
        "reward_head":  RewardHead(latent_dim=latent_dim, n_bins=n_bins, hidden_dim=hidden_dim),
        "continue_head": ContinueHead(latent_dim=latent_dim, hidden_dim=hidden_dim),
        "tau_head":     ValidityHead(latent_dim=latent_dim, action_dim=action_dim,
                                     n_bins=n_bins, hidden_dim=hidden_dim),
        "hazard_head":  HazardHead(latent_dim=latent_dim, n_intervals=n_intervals, hidden_dim=hidden_dim),
        "actor":        Actor(latent_dim=latent_dim, action_dim=action_dim, discrete=False, hidden_dim=hidden_dim),
        "critic":       Critic(latent_dim=latent_dim, n_bins=n_bins, hidden_dim=hidden_dim),
    }


def _make_tiny_buffer(obs_dim: int = 3, action_dim: int = 1, n_steps: int = 256):
    """Fill a small ReplayBuffer with random synthetic data."""
    from bvh_rssm.training.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=n_steps, obs_dim=obs_dim, action_dim=action_dim, seq_len=16)
    rng = np.random.default_rng(42)

    for _ in range(n_steps):
        buf.push(
            obs=rng.standard_normal(obs_dim).astype(np.float32),
            action=rng.standard_normal(action_dim).astype(np.float32),
            reward=float(rng.standard_normal()),
            terminated=bool(rng.integers(0, 2)),
            oracle_tau=int(rng.integers(0, 20)),
            is_interventionist=False,
            rng_state={"dummy": True},
        )
    return buf


class TestTrainPhase3Smoke:
    """Smoke tests: train_phase3() must run without exceptions on a tiny model."""

    def test_phase3_runs_two_steps(self):
        """Instantiate a tiny Trainer, run 2 Phase 3 steps, assert no exception."""
        from bvh_rssm.training.trainer import Trainer, TrainerConfig

        model = _make_tiny_model()
        buf   = _make_tiny_buffer()

        cfg = TrainerConfig(
            phase1_steps=0,
            phase2_steps=0,
            phase3_steps=2,
            learning_rate=1e-3,
            grad_clip=10.0,
            batch_size=4,
            seq_len=16,
            log_every=1,
            checkpoint_every=0,
            device="cpu",
            seed=0,
            run_dir="runs/test_phase3_smoke",
        )

        trainer = Trainer(model, buf, cfg)
        # Should run without raising any exception
        trainer.train_phase3()

    def test_phase3_loss_values_are_finite(self, tmp_path):
        """Verify that loss tensors produced during Phase 3 are all finite."""
        from bvh_rssm.training.trainer import Trainer, TrainerConfig

        model = _make_tiny_model()
        buf   = _make_tiny_buffer()

        logged_metrics: dict = {}

        # Monkey-patch log_metrics to capture values
        import bvh_rssm.training.trainer as trainer_module
        original_log = trainer_module.log_metrics

        def capturing_log(metrics, step):
            logged_metrics.update(metrics)
            original_log(metrics, step)

        trainer_module.log_metrics = capturing_log

        cfg = TrainerConfig(
            phase1_steps=0,
            phase2_steps=0,
            phase3_steps=1,
            learning_rate=1e-3,
            grad_clip=10.0,
            batch_size=4,
            seq_len=16,
            log_every=1,
            checkpoint_every=0,
            device="cpu",
            seed=0,
            run_dir=str(tmp_path / "runs"),
        )

        try:
            trainer = Trainer(model, buf, cfg)
            trainer.train_phase3()
        finally:
            trainer_module.log_metrics = original_log

        # All captured metric values must be finite
        for key, val in logged_metrics.items():
            assert math.isfinite(val), f"Metric '{key}' is not finite: {val}"

    def test_train_dispatches_phase3(self, tmp_path):
        """Trainer.train() must invoke train_phase3() when phase3_steps > 0."""
        from bvh_rssm.training.trainer import Trainer, TrainerConfig

        model = _make_tiny_model()
        buf   = _make_tiny_buffer(n_steps=256)

        cfg = TrainerConfig(
            phase1_steps=1,
            phase2_steps=1,
            phase3_steps=1,
            batch_size=4,
            seq_len=16,
            log_every=1,
            checkpoint_every=0,
            device="cpu",
            seed=0,
            run_dir=str(tmp_path / "runs"),
        )

        phase3_called = []

        trainer = Trainer(model, buf, cfg)
        original_p3 = trainer.train_phase3

        def tracking_p3():
            phase3_called.append(True)
            original_p3()

        trainer.train_phase3 = tracking_p3
        trainer.train()

        assert len(phase3_called) == 1, "train() must call train_phase3() exactly once"
