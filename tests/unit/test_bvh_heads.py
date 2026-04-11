import pytest
import torch
from bvh_rssm.networks.heads import ValidityHead
from bvh_rssm.utils import symlog_bins


class TestValidityHead:
    def setup_method(self):
        self.latent_dim = 48   # h_dim + z_dim
        self.action_dim = 3
        self.n_bins = 64
        self.head = ValidityHead(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            n_bins=self.n_bins,
        )

    def test_logits_shape(self):
        latent = torch.randn(4, self.latent_dim)
        action = torch.randn(4, self.action_dim)
        logits = self.head(latent, action)
        assert logits.shape == (4, self.n_bins)

    def test_decode_shape(self):
        logits = torch.randn(4, self.n_bins)
        tau_hat = self.head.decode(logits)
        assert tau_hat.shape == (4,)

    def test_decode_positive(self):
        """Decoded τ̂ must be non-negative (steps to shift)."""
        logits = torch.randn(4, self.n_bins)
        tau_hat = self.head.decode(logits)
        assert (tau_hat >= 0).all()

    def test_loss_finite(self):
        latent = torch.randn(4, self.latent_dim)
        action = torch.randn(4, self.action_dim)
        oracle_tau = torch.randint(0, 50, (4,)).float()
        loss = self.head.loss(latent, action, oracle_tau)
        assert torch.isfinite(loss)

    def test_loss_differentiable(self):
        latent = torch.randn(4, self.latent_dim, requires_grad=True)
        action = torch.randn(4, self.action_dim)
        oracle_tau = torch.randint(1, 100, (4,)).float()
        loss = self.head.loss(latent, action, oracle_tau)
        loss.backward()
        assert latent.grad is not None
        assert latent.grad.abs().sum() > 0

    def test_stop_grad_detaches_latent(self):
        """With stop_grad=True, gradients must not flow through latent."""
        latent = torch.randn(4, self.latent_dim, requires_grad=True)
        action = torch.randn(4, self.action_dim)
        oracle_tau = torch.ones(4) * 10.0
        loss = self.head.loss(latent, action, oracle_tau, stop_grad=True)
        loss.backward()
        assert latent.grad is None or latent.grad.abs().sum() == 0

    def test_bins_stored_as_buffer(self):
        """bins must be a registered buffer (auto device placement)."""
        assert hasattr(self.head, 'bins')
        assert isinstance(self.head.bins, torch.Tensor)

    def test_loss_nonnegative(self):
        latent = torch.randn(4, self.latent_dim)
        action = torch.randn(4, self.action_dim)
        oracle_tau = torch.ones(4) * 5.0
        loss = self.head.loss(latent, action, oracle_tau)
        assert loss.item() >= 0.0
