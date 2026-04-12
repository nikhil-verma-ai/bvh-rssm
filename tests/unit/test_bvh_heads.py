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
        """bins must be in named_buffers(), not a plain attribute."""
        assert "bins" in dict(self.head.named_buffers()), \
            "bins must be registered via register_buffer, not plain attribute"

    def test_loss_nonnegative(self):
        latent = torch.randn(4, self.latent_dim)
        action = torch.randn(4, self.action_dim)
        oracle_tau = torch.ones(4) * 5.0
        loss = self.head.loss(latent, action, oracle_tau)
        assert loss.item() >= 0.0

    def test_decode_roundtrip(self):
        """Decode of perfect twohot logits must recover target tau within 5%."""
        from bvh_rssm.utils import symlog, twohot
        target_tau = 42.0
        bins = self.head.bins
        target_symlog = symlog(torch.tensor(target_tau))
        twohot_target = twohot(target_symlog.unsqueeze(0), bins)   # [1, n_bins]
        logits = torch.log(twohot_target + 1e-8)                   # [1, n_bins]
        decoded = self.head.decode(logits)                          # [1]
        assert abs(decoded.item() - target_tau) / target_tau < 0.05, \
            f"Expected ~{target_tau}, got {decoded.item()}"

    def test_loss_oracle_tau_zero(self):
        """oracle_tau=0 must not produce NaN or negative loss."""
        latent = torch.randn(2, self.latent_dim)
        action = torch.randn(2, self.action_dim)
        oracle_tau = torch.zeros(2)
        loss = self.head.loss(latent, action, oracle_tau)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0


from bvh_rssm.networks.heads import HazardHead


class TestHazardHead:
    def setup_method(self):
        self.latent_dim = 48
        self.n_intervals = 16
        self.head = HazardHead(latent_dim=self.latent_dim, n_intervals=self.n_intervals)

    def test_hazard_shape_per_source(self):
        """Each source sub-head returns K hazard values in (0, 1)."""
        latent = torch.randn(4, self.latent_dim)
        h_A, h_B, h_C = self.head(latent)
        for h in [h_A, h_B, h_C]:
            assert h.shape == (4, self.n_intervals)
            assert (h > 0).all() and (h < 1).all()

    def test_combined_hazard_shape(self):
        latent = torch.randn(4, self.latent_dim)
        h_total = self.head.combined_hazard(latent)
        assert h_total.shape == (4, self.n_intervals)

    def test_survival_shape(self):
        latent = torch.randn(4, self.latent_dim)
        S = self.head.survival(latent)
        assert S.shape == (4, self.n_intervals)

    def test_survival_decreasing(self):
        """Survival S(t) must be non-increasing: S(t+1) <= S(t)."""
        latent = torch.randn(4, self.latent_dim)
        S = self.head.survival(latent)
        diffs = S[:, 1:] - S[:, :-1]
        assert (diffs <= 0).all(), "Survival function must be non-increasing"

    def test_survival_bounded(self):
        """S(t) must be in (0, 1] for all t."""
        latent = torch.randn(4, self.latent_dim)
        S = self.head.survival(latent)
        assert (S > 0).all() and (S <= 1.0 + 1e-6).all()

    def test_source_b_loss_differentiable(self):
        """Source B is active in Phase 1/2; loss must be differentiable."""
        latent = torch.randn(4, self.latent_dim, requires_grad=True)
        event_times = torch.randint(0, self.n_intervals, (4,))
        event_occurred = torch.ones(4, dtype=torch.bool)
        loss = self.head.loss_source_b(latent, event_times, event_occurred)
        assert torch.isfinite(loss)
        loss.backward()
        assert latent.grad is not None

    def test_gradient_flows_combined(self):
        latent = torch.randn(4, self.latent_dim, requires_grad=True)
        h_total = self.head.combined_hazard(latent)
        h_total.sum().backward()
        assert latent.grad is not None

    def test_n_intervals_stored(self):
        assert self.head.n_intervals == self.n_intervals

    def test_source_ac_near_zero_at_init(self):
        """Sources A and C must produce near-zero hazard at init (sigmoid(-5) ≈ 0.007)."""
        latent = torch.randn(32, self.latent_dim)
        with torch.no_grad():
            h_A, _, h_C = self.head(latent)
        # sigmoid(-5) ≈ 0.0067; allow 3x margin for variance in intermediate activations
        assert h_A.mean().item() < 0.02, f"Source A mean hazard too high: {h_A.mean().item():.4f}"
        assert h_C.mean().item() < 0.02, f"Source C mean hazard too high: {h_C.mean().item():.4f}"
