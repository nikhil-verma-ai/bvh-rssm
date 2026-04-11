import pytest
import torch
import torch.nn.functional as F
from bvh_rssm.utils.distributions import straight_through_sample, sample_categorical


class TestStraightThroughSample:
    def test_output_is_one_hot(self):
        """Forward pass produces valid one-hot vectors."""
        logits = torch.randn(4, 32, 32)  # [batch, n_cats, n_classes]
        out = straight_through_sample(logits)
        # Each of the 32 categoricals should sum to 1
        assert torch.allclose(out.sum(-1), torch.ones(4, 32), atol=1e-5)
        # Each entry should be 0 or 1 in forward
        assert ((out == 0) | (out == 1)).all()

    def test_gradient_flows_through(self):
        """Straight-through: gradient flows back to logits, not blocked."""
        logits = torch.randn(2, 4, 8, requires_grad=True)
        out = straight_through_sample(logits)
        loss = out.sum()
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_output_shape_preserved(self):
        logits = torch.randn(3, 16, 16)
        out = straight_through_sample(logits)
        assert out.shape == logits.shape

    def test_single_batch(self):
        logits = torch.randn(1, 32, 32)
        out = straight_through_sample(logits)
        assert out.shape == (1, 32, 32)


class TestSampleCategorical:
    def test_output_shape(self):
        logits = torch.randn(4, 32)
        out = sample_categorical(logits)
        assert out.shape == (4, 32)

    def test_output_sums_to_one(self):
        logits = torch.randn(4, 32)
        out = sample_categorical(logits)
        assert torch.allclose(out.sum(-1), torch.ones(4), atol=1e-5)

    def test_gradient_flows_through(self):
        logits = torch.randn(2, 8, requires_grad=True)
        out = sample_categorical(logits)
        loss = out.sum()
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_stochastic_different_samples(self):
        """Different calls with the same logits should produce different samples."""
        logits = torch.zeros(100, 4)  # uniform distribution
        samples = [sample_categorical(logits).argmax(-1) for _ in range(5)]
        # Not all samples should be identical (with high probability for uniform)
        all_same = all(torch.equal(samples[0], s) for s in samples[1:])
        assert not all_same
