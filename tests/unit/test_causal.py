"""Tests for bvh_rssm.causal — CausalAttributor and AdaptivePolicyRouter."""
import pytest
import torch
import numpy as np
from bvh_rssm.networks import RSSM, Encoder, ValidityHead
from bvh_rssm.networks.rssm import State
from bvh_rssm.utils.rng import save_rng_state


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B = 3          # batch size
OBS_DIM = 8    # raw observation dim
EMBED_DIM = 16 # encoder output dim = RSSM obs_dim
H_DIM = 32     # RSSM hidden dim
Z_CATS = 4
Z_CLASSES = 4  # z_dim = 16
ACTION_DIM = 2
LATENT_DIM = H_DIM + Z_CATS * Z_CLASSES  # 32 + 16 = 48


@pytest.fixture
def models():
    torch.manual_seed(0)
    rssm = RSSM(
        h_dim=H_DIM,
        z_cats=Z_CATS,
        z_classes=Z_CLASSES,
        obs_dim=EMBED_DIM,
        action_dim=ACTION_DIM,
    )
    encoder = Encoder(obs_dim=OBS_DIM, embed_dim=EMBED_DIM)
    tau_head = ValidityHead(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
    )
    return rssm, encoder, tau_head


@pytest.fixture
def attributor(models):
    from bvh_rssm.causal.attribution import CausalAttributor
    rssm, encoder, tau_head = models
    return CausalAttributor(rssm=rssm, encoder=encoder, tau_head=tau_head)


@pytest.fixture
def rssm_state(models):
    rssm, _, _ = models
    rssm.eval()
    state = rssm.initial_state(batch_size=B)
    # Advance one step so state is non-trivial
    action = torch.randn(B, ACTION_DIM)
    _, state = rssm.imagine(action, state)
    return state


@pytest.fixture
def alt_action():
    return torch.randn(B, ACTION_DIM)


@pytest.fixture
def rng_state():
    return save_rng_state()


# ---------------------------------------------------------------------------
# Task 1: CausalAttributor
# ---------------------------------------------------------------------------

class TestCausalAttributorAssociational:
    def test_returns_tensor(self, attributor, rssm_state, models):
        _, _, tau_head = models
        rssm, _, _ = models
        latent = rssm.get_latent(rssm_state)
        action = torch.randn(B, ACTION_DIM)
        tau_hat = attributor.associational(latent, action)
        assert isinstance(tau_hat, torch.Tensor)

    def test_output_shape(self, attributor, rssm_state, models):
        rssm, _, _ = models
        latent = rssm.get_latent(rssm_state)
        action = torch.randn(B, ACTION_DIM)
        tau_hat = attributor.associational(latent, action)
        assert tau_hat.shape == (B,), f"Expected ({B},), got {tau_hat.shape}"

    def test_output_finite(self, attributor, rssm_state, models):
        rssm, _, _ = models
        latent = rssm.get_latent(rssm_state)
        action = torch.randn(B, ACTION_DIM)
        tau_hat = attributor.associational(latent, action)
        assert torch.isfinite(tau_hat).all(), "associational tau_hat must be finite"

    def test_output_non_negative(self, attributor, rssm_state, models):
        rssm, _, _ = models
        latent = rssm.get_latent(rssm_state)
        action = torch.randn(B, ACTION_DIM)
        tau_hat = attributor.associational(latent, action)
        assert (tau_hat >= 0).all(), "decoded tau_hat must be >= 0"

    def test_matches_direct_head_call(self, attributor, rssm_state, models):
        """associational() must produce identical output to tau_head.decode(tau_head(latent, action))."""
        rssm, _, tau_head = models
        latent = rssm.get_latent(rssm_state)
        action = torch.randn(B, ACTION_DIM)
        tau_hat_attributor = attributor.associational(latent, action)
        logits = tau_head(latent, action, stop_grad=False)
        tau_hat_direct = tau_head.decode(logits)
        assert torch.allclose(tau_hat_attributor, tau_hat_direct), (
            "associational must equal direct tau_head.decode(tau_head(latent, action))"
        )


class TestCausalAttributorInterventional:
    def test_returns_tensor(self, attributor, rssm_state):
        alt_action = torch.randn(B, ACTION_DIM)
        tau_hat = attributor.interventional(rssm_state, alt_action)
        assert isinstance(tau_hat, torch.Tensor)

    def test_output_shape(self, attributor, rssm_state):
        alt_action = torch.randn(B, ACTION_DIM)
        tau_hat = attributor.interventional(rssm_state, alt_action)
        assert tau_hat.shape == (B,), f"Expected ({B},), got {tau_hat.shape}"

    def test_output_finite(self, attributor, rssm_state):
        alt_action = torch.randn(B, ACTION_DIM)
        tau_hat = attributor.interventional(rssm_state, alt_action)
        assert torch.isfinite(tau_hat).all(), "interventional tau_hat must be finite"

    def test_output_non_negative(self, attributor, rssm_state):
        alt_action = torch.randn(B, ACTION_DIM)
        tau_hat = attributor.interventional(rssm_state, alt_action)
        assert (tau_hat >= 0).all(), "decoded tau_hat must be >= 0"

    def test_different_action_different_output(self, attributor, rssm_state, models):
        """Two distinct alt_actions must (with overwhelming probability) produce different tau_hats."""
        rssm, _, _ = models
        rssm.train()  # stochastic mode makes divergence nearly certain
        torch.manual_seed(42)
        a1 = torch.randn(B, ACTION_DIM)
        a2 = torch.randn(B, ACTION_DIM) * 5.0  # large offset to ensure real difference
        tau1 = attributor.interventional(rssm_state, a1)
        tau2 = attributor.interventional(rssm_state, a2)
        # At least one batch element must differ
        assert not torch.allclose(tau1, tau2), (
            "Interventional outputs must differ for distinct actions"
        )

    def test_does_not_mutate_rssm_state(self, attributor, rssm_state):
        """interventional() must not modify rssm_state in place."""
        h_before = rssm_state.h.clone()
        z_before = rssm_state.z.clone()
        alt_action = torch.randn(B, ACTION_DIM)
        attributor.interventional(rssm_state, alt_action)
        assert torch.allclose(rssm_state.h, h_before), "rssm_state.h mutated"
        assert torch.allclose(rssm_state.z, z_before), "rssm_state.z mutated"


class TestCausalAttributorCounterfactual:
    def test_returns_tensor(self, attributor, rssm_state):
        alt_action = torch.randn(B, ACTION_DIM)
        rng_state = save_rng_state()
        tau_hat = attributor.counterfactual(rssm_state, alt_action, rng_state)
        assert isinstance(tau_hat, torch.Tensor)

    def test_output_shape(self, attributor, rssm_state):
        alt_action = torch.randn(B, ACTION_DIM)
        rng_state = save_rng_state()
        tau_hat = attributor.counterfactual(rssm_state, alt_action, rng_state)
        assert tau_hat.shape == (B,), f"Expected ({B},), got {tau_hat.shape}"

    def test_output_finite(self, attributor, rssm_state):
        alt_action = torch.randn(B, ACTION_DIM)
        rng_state = save_rng_state()
        tau_hat = attributor.counterfactual(rssm_state, alt_action, rng_state)
        assert torch.isfinite(tau_hat).all(), "counterfactual tau_hat must be finite"

    def test_output_non_negative(self, attributor, rssm_state):
        alt_action = torch.randn(B, ACTION_DIM)
        rng_state = save_rng_state()
        tau_hat = attributor.counterfactual(rssm_state, alt_action, rng_state)
        assert (tau_hat >= 0).all(), "decoded tau_hat must be >= 0"

    def test_same_rng_same_action_matches_interventional(self, attributor, rssm_state, models):
        """Counterfactual with the *same* action as interventional and restored RNG
        must produce *identical* tau_hat as interventional under eval mode
        (eval mode is deterministic — argmax not sample — so RNG doesn't matter).
        Under eval mode, interventional and counterfactual with the same action
        must agree exactly."""
        rssm, _, _ = models
        rssm.eval()
        alt_action = torch.randn(B, ACTION_DIM)
        rng_state = save_rng_state()
        tau_cf = attributor.counterfactual(rssm_state, alt_action, rng_state)
        tau_iv = attributor.interventional(rssm_state, alt_action)
        assert torch.allclose(tau_cf, tau_iv, atol=1e-6), (
            "Under eval (deterministic), counterfactual and interventional with "
            "the same action must produce identical tau_hat"
        )

    def test_rng_restored_after_call(self, attributor, rssm_state, alt_action, rng_state):
        """RNG must be fully restored after counterfactual() returns.

        Force the RSSM into training mode so imagine() calls _sample_z and
        actually consumes RNG. After counterfactual(), the global RNG must
        be back to where it was before the call.
        """
        import torch
        # Force training mode so _sample_z samples (eval uses argmax)
        attributor.rssm.train()
        try:
            pre_call_rng = torch.get_rng_state()
            attributor.counterfactual(rssm_state, alt_action, rng_state)
            post_call_rng = torch.get_rng_state()
            assert torch.equal(pre_call_rng, post_call_rng), (
                "RNG state must be identical before and after counterfactual() call"
            )
        finally:
            attributor.rssm.eval()

    def test_does_not_mutate_rssm_state(self, attributor, rssm_state):
        """counterfactual() must not modify rssm_state in place."""
        h_before = rssm_state.h.clone()
        z_before = rssm_state.z.clone()
        alt_action = torch.randn(B, ACTION_DIM)
        rng_state = save_rng_state()
        attributor.counterfactual(rssm_state, alt_action, rng_state)
        assert torch.allclose(rssm_state.h, h_before), "rssm_state.h mutated"
        assert torch.allclose(rssm_state.z, z_before), "rssm_state.z mutated"


# ---------------------------------------------------------------------------
# Task 2: AdaptivePolicyRouter
# ---------------------------------------------------------------------------

class TestThresholdsFromSurvival:
    @pytest.fixture
    def router(self):
        from bvh_rssm.causal.router import AdaptivePolicyRouter
        return AdaptivePolicyRouter()

    def test_tau_hi_is_first_index_at_or_below_080(self, router):
        # S drops from 0.95 at t=0 to 0.75 at t=3 — tau_hi should be 3
        # (first index where S <= 0.80)
        S = torch.tensor([0.95, 0.90, 0.82, 0.75, 0.50, 0.25, 0.10, 0.05])
        tau_hi, tau_min = router.thresholds_from_survival(S)
        assert tau_hi == 3, f"Expected tau_hi=3, got {tau_hi}"

    def test_tau_min_is_first_index_at_or_below_020(self, router):
        # S drops to 0.15 at t=6
        S = torch.tensor([0.95, 0.90, 0.82, 0.75, 0.50, 0.25, 0.15, 0.05])
        tau_hi, tau_min = router.thresholds_from_survival(S)
        assert tau_min == 6, f"Expected tau_min=6, got {tau_min}"

    def test_tau_hi_exact_boundary(self, router):
        # S is exactly 0.80 at index 2 — should be included (<=)
        S = torch.tensor([0.95, 0.85, 0.80, 0.70, 0.50, 0.30, 0.15, 0.05])
        tau_hi, _ = router.thresholds_from_survival(S)
        assert tau_hi == 2, f"Expected tau_hi=2 at exact 0.80 boundary, got {tau_hi}"

    def test_tau_min_exact_boundary(self, router):
        # S is exactly 0.20 at index 5
        S = torch.tensor([0.95, 0.85, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05])
        _, tau_min = router.thresholds_from_survival(S)
        assert tau_min == 5, f"Expected tau_min=5 at exact 0.20 boundary, got {tau_min}"

    def test_fallback_when_S_never_crosses_080(self, router):
        # S stays above 0.80 for all K — tau_hi should be K-1
        S = torch.tensor([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92])
        K = S.shape[0]
        tau_hi, _ = router.thresholds_from_survival(S)
        assert tau_hi == K - 1, f"Expected fallback tau_hi={K-1}, got {tau_hi}"

    def test_fallback_when_S_never_crosses_020(self, router):
        # S stays above 0.20 for all K — tau_min should be K-1
        S = torch.tensor([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25])
        K = S.shape[0]
        _, tau_min = router.thresholds_from_survival(S)
        assert tau_min == K - 1, f"Expected fallback tau_min={K-1}, got {tau_min}"

    def test_returns_integers(self, router):
        S = torch.tensor([0.95, 0.85, 0.75, 0.50, 0.25, 0.10, 0.05, 0.02])
        tau_hi, tau_min = router.thresholds_from_survival(S)
        assert isinstance(tau_hi, int), f"tau_hi must be int, got {type(tau_hi)}"
        assert isinstance(tau_min, int), f"tau_min must be int, got {type(tau_min)}"

    def test_tau_hi_leq_tau_min_for_well_shaped_curve(self, router):
        # In a properly monotone-decreasing survival curve, tau_hi <= tau_min
        # (80% threshold is crossed before 20% threshold)
        S = torch.tensor([0.95, 0.85, 0.75, 0.50, 0.25, 0.10, 0.05, 0.02])
        tau_hi, tau_min = router.thresholds_from_survival(S)
        assert tau_hi <= tau_min, (
            f"For monotone S, tau_hi ({tau_hi}) should be <= tau_min ({tau_min})"
        )


class TestClassify:
    @pytest.fixture
    def router(self):
        from bvh_rssm.causal.router import AdaptivePolicyRouter
        return AdaptivePolicyRouter()

    @pytest.fixture
    def typical_S(self):
        # tau_hi=2 (S drops to 0.75 <= 0.80), tau_min=5 (S drops to 0.15 <= 0.20)
        return torch.tensor([0.95, 0.85, 0.75, 0.50, 0.25, 0.15, 0.08, 0.03])

    def test_classify_high_when_above_tau_hi(self, router, typical_S):
        from bvh_rssm.causal.router import RouterState
        # tau_min = 5, so tau_hat > 5 is HIGH (HIGH requires exceeding the pessimistic threshold)
        state = router.classify(tau_hat=6.0, S=typical_S)
        assert state == RouterState.HIGH, f"Expected HIGH, got {state}"

    def test_classify_stale_when_below_tau_hi(self, router, typical_S):
        from bvh_rssm.causal.router import RouterState
        # tau_hi = 2 (optimistic crossing); tau_hat < tau_hi is STALE
        state = router.classify(tau_hat=1.0, S=typical_S)
        assert state == RouterState.STALE, f"Expected STALE, got {state}"

    def test_classify_dim_at_tau_hi(self, router, typical_S):
        from bvh_rssm.causal.router import RouterState
        # tau_hat == tau_hi is DIM (inclusive)
        state = router.classify(tau_hat=2.0, S=typical_S)
        assert state == RouterState.DIM, f"Expected DIM at tau_hi boundary, got {state}"

    def test_classify_dim_at_tau_min(self, router, typical_S):
        from bvh_rssm.causal.router import RouterState
        # tau_hat == tau_min is DIM (inclusive)
        state = router.classify(tau_hat=5.0, S=typical_S)
        # tau_hi=2, tau_min=5: tau_hat=5 is within [tau_hi, tau_min] so DIM
        state = router.classify(tau_hat=5.0, S=typical_S)
        assert state == RouterState.DIM, f"Expected DIM at tau_min boundary, got {state}"

    def test_classify_high_strictly_above_tau_min(self, router, typical_S):
        from bvh_rssm.causal.router import RouterState
        # tau_hat = 5.1 > tau_min (5) → HIGH (HIGH requires exceeding the pessimistic threshold)
        state = router.classify(tau_hat=5.1, S=typical_S)
        assert state == RouterState.HIGH, f"Expected HIGH for tau_hat > tau_min, got {state}"

    def test_classify_stale_strictly_below_tau_hi(self, router, typical_S):
        from bvh_rssm.causal.router import RouterState
        # tau_hat = 1.9 < tau_hi (2) → STALE (strictly below the optimistic threshold)
        state = router.classify(tau_hat=1.9, S=typical_S)
        assert state == RouterState.STALE, f"Expected STALE for tau_hat < tau_hi, got {state}"

    def test_classify_dim_in_middle(self, router, typical_S):
        from bvh_rssm.causal.router import RouterState
        # tau_hat = 3.5 is between tau_hi (2) and tau_min (5) → DIM
        state = router.classify(tau_hat=3.5, S=typical_S)
        assert state == RouterState.DIM, f"Expected DIM for 2 < tau_hat < 5, got {state}"

    def test_all_three_states_reachable(self, router, typical_S):
        from bvh_rssm.causal.router import RouterState
        states = {
            router.classify(tau_hat=10.0, S=typical_S),  # HIGH
            router.classify(tau_hat=3.5, S=typical_S),   # DIM
            router.classify(tau_hat=1.0, S=typical_S),   # STALE
        }
        assert RouterState.HIGH in states
        assert RouterState.DIM in states
        assert RouterState.STALE in states


class TestImaginationHorizon:
    @pytest.fixture
    def router(self):
        from bvh_rssm.causal.router import AdaptivePolicyRouter
        return AdaptivePolicyRouter()

    def test_high_returns_full_horizon(self, router):
        from bvh_rssm.causal.router import RouterState
        assert router.imagination_horizon(RouterState.HIGH, tau_hat=12.0, full_horizon=16) == 16

    def test_high_respects_custom_full_horizon(self, router):
        from bvh_rssm.causal.router import RouterState
        assert router.imagination_horizon(RouterState.HIGH, tau_hat=5.0, full_horizon=32) == 32

    def test_stale_returns_one(self, router):
        from bvh_rssm.causal.router import RouterState
        assert router.imagination_horizon(RouterState.STALE, tau_hat=0.5, full_horizon=16) == 1

    def test_stale_always_returns_one_regardless_of_tau(self, router):
        from bvh_rssm.causal.router import RouterState
        for tau in [0.0, 0.5, 1.0, 2.0, 100.0]:
            result = router.imagination_horizon(RouterState.STALE, tau_hat=tau, full_horizon=16)
            assert result == 1, f"STALE must return 1 for tau_hat={tau}, got {result}"

    def test_dim_is_half_tau_hat_int(self, router):
        from bvh_rssm.causal.router import RouterState
        # tau_hat=6.0 → int(6.0/2) = 3
        assert router.imagination_horizon(RouterState.DIM, tau_hat=6.0, full_horizon=16) == 3

    def test_dim_floors_correctly(self, router):
        from bvh_rssm.causal.router import RouterState
        # tau_hat=7.0 → int(7.0/2) = 3
        assert router.imagination_horizon(RouterState.DIM, tau_hat=7.0, full_horizon=16) == 3

    def test_dim_minimum_is_one(self, router):
        from bvh_rssm.causal.router import RouterState
        # tau_hat=0.0 → max(1, int(0/2)) = max(1, 0) = 1
        assert router.imagination_horizon(RouterState.DIM, tau_hat=0.0, full_horizon=16) == 1

    def test_dim_minimum_for_small_tau_hat(self, router):
        from bvh_rssm.causal.router import RouterState
        # tau_hat=1.5 → int(1.5/2) = 0 → max(1, 0) = 1
        assert router.imagination_horizon(RouterState.DIM, tau_hat=1.5, full_horizon=16) == 1

    def test_dim_large_tau_hat(self, router):
        from bvh_rssm.causal.router import RouterState
        # tau_hat=20.0 → int(20/2) = 10
        assert router.imagination_horizon(RouterState.DIM, tau_hat=20.0, full_horizon=16) == 10

    def test_return_type_is_int(self, router):
        from bvh_rssm.causal.router import RouterState
        result = router.imagination_horizon(RouterState.HIGH, tau_hat=5.0, full_horizon=16)
        assert isinstance(result, int), f"Expected int, got {type(result)}"


# ---------------------------------------------------------------------------
# Task 3: Integration smoke test
# ---------------------------------------------------------------------------

class TestIntegrationSmoke:
    """Full pipeline: models → CausalAttributor → AdaptivePolicyRouter.

    Exercises the import path from bvh_rssm.causal (not the submodule paths)
    and verifies that all three Pearl levels + router compose without error.
    """

    def test_top_level_imports(self):
        """All three public names must be importable from bvh_rssm.causal."""
        from bvh_rssm.causal import CausalAttributor, AdaptivePolicyRouter, RouterState
        assert CausalAttributor is not None
        assert AdaptivePolicyRouter is not None
        assert RouterState is not None

    def test_full_pipeline_produces_finite_results(self):
        """Build models, run all three levels, classify, get horizon."""
        from bvh_rssm.causal import CausalAttributor, AdaptivePolicyRouter, RouterState

        torch.manual_seed(7)
        B_smoke = 2
        OBS = 6
        EMBED = 12
        H = 24
        ZCATS, ZCLASSES = 4, 4
        ACT = 2
        LATENT = H + ZCATS * ZCLASSES  # 24 + 16 = 40

        rssm = RSSM(h_dim=H, z_cats=ZCATS, z_classes=ZCLASSES,
                    obs_dim=EMBED, action_dim=ACT)
        encoder = Encoder(obs_dim=OBS, embed_dim=EMBED)
        tau_head = ValidityHead(latent_dim=LATENT, action_dim=ACT)
        from bvh_rssm.networks import HazardHead
        hazard_head = HazardHead(latent_dim=LATENT, n_intervals=16)

        attributor = CausalAttributor(rssm=rssm, encoder=encoder, tau_head=tau_head)
        router = AdaptivePolicyRouter()

        rssm.eval()
        state = rssm.initial_state(batch_size=B_smoke)
        action = torch.randn(B_smoke, ACT)
        _, state = rssm.imagine(action, state)

        latent = rssm.get_latent(state)

        with torch.no_grad():
            # Level 1: associational
            tau_l1 = attributor.associational(latent, action)
            assert tau_l1.shape == (B_smoke,)
            assert torch.isfinite(tau_l1).all()

            # Level 2: interventional
            alt_action = torch.randn(B_smoke, ACT)
            tau_l2 = attributor.interventional(state, alt_action)
            assert tau_l2.shape == (B_smoke,)
            assert torch.isfinite(tau_l2).all()

            # Level 3: counterfactual — need RNG captured before factual imagine step
            rng_pre_imagine = save_rng_state()
            _, _ = rssm.imagine(action, state)  # factual imagine (advances RNG)
            tau_l3 = attributor.counterfactual(state, alt_action, rng_pre_imagine)
            assert tau_l3.shape == (B_smoke,)
            assert torch.isfinite(tau_l3).all()

            # Router: classify sample 0 using its survival curve
            S_batch = hazard_head.survival(latent)   # [B_smoke, 16]
            S_0 = S_batch[0]                          # [16]
            tau_scalar = tau_l1[0].item()
            router_state = router.classify(tau_hat=tau_scalar, S=S_0)
            assert router_state in (RouterState.HIGH, RouterState.DIM, RouterState.STALE)

            horizon = router.imagination_horizon(router_state, tau_hat=tau_scalar, full_horizon=16)
            assert isinstance(horizon, int)
            assert 1 <= horizon <= 16
