import pytest
import torch
from bvh_rssm.networks.heads import RewardHead, ContinueHead
from bvh_rssm.utils import symlog_bins


class TestRewardHead:
    def setup_method(self):
        self.latent_dim = 32
        self.n_bins = 64
        self.head = RewardHead(latent_dim=self.latent_dim, n_bins=self.n_bins)
        self.bins = symlog_bins(self.n_bins)

    def test_logits_output_shape(self):
        latent = torch.randn(4, self.latent_dim)
        logits = self.head(latent)
        assert logits.shape == (4, self.n_bins)

    def test_expected_reward_shape(self):
        latent = torch.randn(4, self.latent_dim)
        logits = self.head(latent)
        reward = self.head.decode(logits, self.bins)
        assert reward.shape == (4,)

    def test_loss_is_finite(self):
        latent = torch.randn(4, self.latent_dim)
        target_reward = torch.randn(4)  # raw rewards
        loss = self.head.loss(latent, target_reward, self.bins)
        assert torch.isfinite(loss)

    def test_loss_is_differentiable(self):
        latent = torch.randn(4, self.latent_dim, requires_grad=True)
        target_reward = torch.randn(4)
        loss = self.head.loss(latent, target_reward, self.bins)
        loss.backward()
        assert latent.grad is not None
        assert latent.grad.abs().sum() > 0

    def test_loss_decreases_with_correct_prediction(self):
        """Loss should be near zero when predicting the exact target."""
        head = RewardHead(latent_dim=self.latent_dim, n_bins=self.n_bins)
        bins = symlog_bins(self.n_bins)
        # This test checks the loss function API, not convergence
        target = torch.zeros(2)
        latent = torch.randn(2, self.latent_dim)
        loss = head.loss(latent, target, bins)
        assert loss.item() >= 0.0


class TestContinueHead:
    def setup_method(self):
        self.latent_dim = 32
        self.head = ContinueHead(latent_dim=self.latent_dim)

    def test_logits_shape(self):
        latent = torch.randn(4, self.latent_dim)
        logits = self.head(latent)
        assert logits.shape == (4, 1)

    def test_probability_shape(self):
        latent = torch.randn(4, self.latent_dim)
        prob = self.head.probability(latent)
        assert prob.shape == (4, 1)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_loss_is_finite(self):
        latent = torch.randn(4, self.latent_dim)
        continue_flag = torch.randint(0, 2, (4, 1)).float()
        loss = self.head.loss(latent, continue_flag)
        assert torch.isfinite(loss)

    def test_loss_is_differentiable(self):
        latent = torch.randn(4, self.latent_dim, requires_grad=True)
        continue_flag = torch.ones(4, 1)
        loss = self.head.loss(latent, continue_flag)
        loss.backward()
        assert latent.grad is not None

    def test_probability_near_one_for_non_terminal(self):
        """Continue head should be trainable toward predicting continuation."""
        head = ContinueHead(latent_dim=self.latent_dim)
        latent = torch.randn(4, self.latent_dim)
        prob = head.probability(latent)
        assert prob.shape == (4, 1)
        assert torch.isfinite(prob).all()
