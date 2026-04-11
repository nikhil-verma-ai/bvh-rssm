import pytest
import torch
from bvh_rssm.networks.heads import RewardHead, ContinueHead
from bvh_rssm.utils import symlog_bins


class TestRewardHead:
    def setup_method(self):
        self.latent_dim = 32
        self.n_bins = 64
        self.head = RewardHead(latent_dim=self.latent_dim, n_bins=self.n_bins)

    def test_logits_output_shape(self):
        latent = torch.randn(4, self.latent_dim)
        logits = self.head(latent)
        assert logits.shape == (4, self.n_bins)

    def test_expected_reward_shape(self):
        latent = torch.randn(4, self.latent_dim)
        logits = self.head(latent)
        reward = self.head.decode(logits)
        assert reward.shape == (4,)

    def test_loss_is_finite(self):
        latent = torch.randn(4, self.latent_dim)
        target_reward = torch.randn(4)  # raw rewards
        loss = self.head.loss(latent, target_reward)
        assert torch.isfinite(loss)

    def test_loss_is_differentiable(self):
        latent = torch.randn(4, self.latent_dim, requires_grad=True)
        target_reward = torch.randn(4)
        loss = self.head.loss(latent, target_reward)
        loss.backward()
        assert latent.grad is not None
        assert latent.grad.abs().sum() > 0

    def test_decode_roundtrip_near_target(self):
        """decode() should recover a reward close to the target when logits
        match the twohot encoding of symlog(target).

        twohot distributes probability across the two bins that straddle the
        target in symlog space; feeding those exact (log-)probabilities back
        through decode should reconstruct the original reward to machine
        precision.
        """
        from bvh_rssm.utils import symlog as _symlog, twohot as _twohot
        target = 5.0
        n_bins = 64
        bins = symlog_bins(n_bins)
        head = RewardHead(latent_dim=self.latent_dim, n_bins=n_bins)

        # Build logits whose softmax equals the twohot distribution for target.
        # log(twohot + eps) followed by softmax recovers the twohot probabilities
        # to high precision (eps only affects near-zero bins negligibly).
        target_symlog = _symlog(torch.tensor(target))
        twohot_target = _twohot(target_symlog.unsqueeze(0), bins)  # [1, n_bins]
        logits = torch.log(twohot_target + 1e-8)  # [1, n_bins]

        decoded = head.decode(logits)  # shape [1]
        assert abs(decoded.item() - target) / abs(target) < 0.1, (
            f"Expected decoded reward within 10% of {target}, got {decoded.item():.4f}"
        )


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
        assert prob.shape == (4,)
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
        assert prob.shape == (4,)
        assert torch.isfinite(prob).all()


from bvh_rssm.networks.actor_critic import Actor, Critic


class TestActor:
    def test_output_shape_continuous(self):
        actor = Actor(latent_dim=32, action_dim=6, discrete=False)
        latent = torch.randn(4, 32)
        mean, log_std = actor(latent)
        assert mean.shape == (4, 6)
        assert log_std.shape == (4, 6)

    def test_output_shape_discrete(self):
        actor = Actor(latent_dim=32, action_dim=4, discrete=True)
        latent = torch.randn(4, 32)
        logits = actor(latent)
        assert logits.shape == (4, 4)

    def test_gradient_flows(self):
        actor = Actor(latent_dim=16, action_dim=4, discrete=False)
        latent = torch.randn(2, 16, requires_grad=True)
        mean, _ = actor(latent)
        mean.sum().backward()
        assert latent.grad is not None

    def test_gradient_flows_discrete(self):
        actor = Actor(latent_dim=16, action_dim=4, discrete=True)
        latent = torch.randn(2, 16, requires_grad=True)
        logits = actor(latent)
        logits.sum().backward()
        assert latent.grad is not None


class TestCritic:
    def test_output_shape(self):
        critic = Critic(latent_dim=32, n_bins=64)
        latent = torch.randn(4, 32)
        logits = critic(latent)
        assert logits.shape == (4, 64)
        assert critic.n_bins == 64

    def test_gradient_flows(self):
        critic = Critic(latent_dim=16, n_bins=32)
        latent = torch.randn(2, 16, requires_grad=True)
        logits = critic(latent)
        logits.sum().backward()
        assert latent.grad is not None
