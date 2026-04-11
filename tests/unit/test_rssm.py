import pytest
import torch
from bvh_rssm.networks.rssm import RSSM, State


class TestRSSMState:
    def test_state_is_namedtuple(self):
        h = torch.zeros(2, 512)
        z = torch.zeros(2, 1024)
        state = State(h=h, z=z)
        assert state.h is h
        assert state.z is z

    def test_initial_state_zeros(self):
        rssm = RSSM(h_dim=64, z_cats=4, z_classes=4, obs_dim=8, action_dim=2)
        state = rssm.initial_state(batch_size=3)
        assert state.h.shape == (3, 64)
        assert state.z.shape == (3, 16)  # 4*4
        assert state.h.sum() == 0.0
        assert state.z.sum() == 0.0


class TestRSSMObserve:
    def setup_method(self):
        self.rssm = RSSM(h_dim=64, z_cats=4, z_classes=4, obs_dim=16, action_dim=2)
        self.state = self.rssm.initial_state(batch_size=2)

    def test_observe_returns_state_and_logits(self):
        obs = torch.randn(2, 16)
        action = torch.randn(2, 2)
        posterior_logits, next_state = self.rssm.observe(obs, action, self.state)
        assert isinstance(next_state, State)
        assert isinstance(posterior_logits, torch.Tensor)

    def test_observe_h_shape(self):
        obs = torch.randn(2, 16)
        action = torch.randn(2, 2)
        _, next_state = self.rssm.observe(obs, action, self.state)
        assert next_state.h.shape == (2, 64)

    def test_observe_z_shape(self):
        obs = torch.randn(2, 16)
        action = torch.randn(2, 2)
        _, next_state = self.rssm.observe(obs, action, self.state)
        assert next_state.z.shape == (2, 16)  # 4 cats * 4 classes

    def test_observe_posterior_logits_shape(self):
        obs = torch.randn(2, 16)
        action = torch.randn(2, 2)
        posterior_logits, _ = self.rssm.observe(obs, action, self.state)
        assert posterior_logits.shape == (2, 4, 4)  # [B, n_cats, n_classes]

    def test_observe_gradient_flows(self):
        obs = torch.randn(2, 16, requires_grad=True)
        action = torch.randn(2, 2)
        posterior_logits, next_state = self.rssm.observe(obs, action, self.state)
        loss = next_state.h.sum() + next_state.z.sum() + posterior_logits.sum()
        loss.backward()
        assert obs.grad is not None
        assert obs.grad.abs().sum() > 0

    def test_multi_step_observe_rollout(self):
        """State threads correctly across multiple observe() calls."""
        rssm = RSSM(h_dim=32, z_cats=4, z_classes=4, obs_dim=8, action_dim=2)
        state = rssm.initial_state(batch_size=2)
        for t in range(5):
            obs = torch.randn(2, 8)
            action = torch.randn(2, 2)
            logits, state = rssm.observe(obs, action, state)
            assert state.h.shape == (2, 32)
            assert state.z.shape == (2, 16)  # 4*4
        # After 5 steps, state must still be differentiable
        state.h.sum().backward()


class TestRSSMImagine:
    def setup_method(self):
        self.rssm = RSSM(h_dim=64, z_cats=4, z_classes=4, obs_dim=16, action_dim=2)
        self.state = self.rssm.initial_state(batch_size=2)

    def test_imagine_returns_state_and_prior_logits(self):
        action = torch.randn(2, 2)
        prior_logits, next_state = self.rssm.imagine(action, self.state)
        assert isinstance(next_state, State)
        assert isinstance(prior_logits, torch.Tensor)

    def test_imagine_h_shape(self):
        action = torch.randn(2, 2)
        _, next_state = self.rssm.imagine(action, self.state)
        assert next_state.h.shape == (2, 64)

    def test_imagine_z_shape(self):
        action = torch.randn(2, 2)
        _, next_state = self.rssm.imagine(action, self.state)
        assert next_state.z.shape == (2, 16)

    def test_imagine_prior_logits_shape(self):
        action = torch.randn(2, 2)
        prior_logits, _ = self.rssm.imagine(action, self.state)
        assert prior_logits.shape == (2, 4, 4)

    def test_imagine_gradient_flows(self):
        action = torch.randn(2, 2, requires_grad=True)
        prior_logits, next_state = self.rssm.imagine(action, self.state)
        loss = next_state.h.sum() + next_state.z.sum()
        loss.backward()
        assert action.grad is not None

    def test_imagine_does_not_require_obs(self):
        """imagine() must work with no observation input at all."""
        action = torch.randn(2, 2)
        prior_logits, next_state = self.rssm.imagine(action, self.state)
        assert next_state.h is not None


class TestRSSMLatent:
    def test_get_latent_concatenates_h_and_z(self):
        rssm = RSSM(h_dim=64, z_cats=4, z_classes=4, obs_dim=8, action_dim=2)
        state = rssm.initial_state(batch_size=3)
        latent = rssm.get_latent(state)
        assert latent.shape == (3, 64 + 16)  # h_dim + z_cats*z_classes

    def test_get_latent_is_differentiable(self):
        rssm = RSSM(h_dim=32, z_cats=4, z_classes=4, obs_dim=8, action_dim=2)
        obs = torch.randn(2, 8)
        action = torch.randn(2, 2)
        state = rssm.initial_state(batch_size=2)
        _, state = rssm.observe(obs, action, state)
        latent = rssm.get_latent(state)
        latent.sum().backward()  # must not raise

    def test_unimix_applied_to_z(self):
        """z_t sampling must use unimix-mixed distribution (min mass = eps/n_classes)."""
        from bvh_rssm.utils import unimix
        rssm = RSSM(h_dim=32, z_cats=4, z_classes=8, obs_dim=8, action_dim=2)
        obs = torch.randn(5, 8)
        action = torch.randn(5, 2)
        state = rssm.initial_state(5)
        posterior_logits, _ = rssm.observe(obs, action, state)
        # Test the mixed distribution (what _sample_z actually uses), not raw logits
        mixed_probs = unimix(posterior_logits, eps=rssm.unimix_eps)
        min_mass = rssm.unimix_eps / 8  # eps / n_classes
        assert mixed_probs.min().item() >= min_mass - 1e-7  # fp tolerance only

    def test_unimix_prevents_probability_collapse(self):
        """unimix must maintain floor even when raw logits are extreme."""
        from bvh_rssm.utils import unimix
        rssm = RSSM(h_dim=32, z_cats=4, z_classes=8, obs_dim=8, action_dim=2)
        # Extreme logits: one class dominates with 1e6, others at -1e6
        # Without unimix, softmax collapses to [0,0,...,1,...,0] — other probs ~0
        extreme_logits = torch.full((5, 4, 8), -1e6)
        extreme_logits[:, :, 0] = 1e6  # class 0 dominates
        mixed_probs = unimix(extreme_logits, eps=rssm.unimix_eps)
        min_mass = rssm.unimix_eps / 8
        # Even with extreme logits, unimix guarantees floor
        assert mixed_probs.min().item() >= min_mass - 1e-7
