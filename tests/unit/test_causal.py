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
