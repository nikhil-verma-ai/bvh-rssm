import pytest
import torch
from bvh_rssm.utils.math import symlog, symexp, twohot, twohot_decode, symlog_bins, unimix


class TestSymlog:
    def test_zero(self):
        assert symlog(torch.tensor(0.0)).item() == pytest.approx(0.0)

    def test_positive(self):
        x = torch.tensor(1.0)
        # sign(1) * log(|1| + 1) = log(2)
        expected = torch.log(torch.tensor(2.0)).item()
        assert symlog(x).item() == pytest.approx(expected, rel=1e-5)

    def test_negative(self):
        x = torch.tensor(-1.0)
        expected = -torch.log(torch.tensor(2.0)).item()
        assert symlog(x).item() == pytest.approx(expected, rel=1e-5)

    def test_large_value_compressed(self):
        x = torch.tensor(1000.0)
        assert symlog(x).item() < 10.0

    def test_batch(self):
        x = torch.tensor([-5.0, 0.0, 5.0])
        out = symlog(x)
        assert out.shape == (3,)


class TestSymexp:
    def test_zero(self):
        assert symexp(torch.tensor(0.0)).item() == pytest.approx(0.0)

    def test_inverse_of_symlog(self):
        x = torch.linspace(-10.0, 10.0, 100)
        reconstructed = symexp(symlog(x))
        assert torch.allclose(reconstructed, x, atol=1e-5)

    def test_symlog_inverse_of_symexp(self):
        y = torch.linspace(-5.0, 5.0, 50)
        reconstructed = symlog(symexp(y))
        assert torch.allclose(reconstructed, y, atol=1e-5)


class TestTwohot:
    def test_output_shape(self):
        bins = symlog_bins(255)
        x = torch.tensor([0.5, -1.2, 3.7])
        out = twohot(x, bins)
        assert out.shape == (3, 255)

    def test_sums_to_one(self):
        bins = symlog_bins(255)
        x = torch.linspace(-5.0, 5.0, 20)
        out = twohot(x, bins)
        assert torch.allclose(out.sum(-1), torch.ones(20), atol=1e-5)

    def test_non_negative(self):
        bins = symlog_bins(255)
        x = torch.randn(10)
        out = twohot(x, bins)
        assert (out >= 0).all()

    def test_exactly_two_nonzero_entries(self):
        bins = symlog_bins(255)
        x = torch.tensor([0.123, -0.456, 2.789])
        out = twohot(x, bins)
        nonzero_counts = (out > 0).sum(-1)
        assert (nonzero_counts == 2).all()

    def test_decode_roundtrip_within_bin_width(self):
        bins = symlog_bins(255)
        bin_width = (bins[-1] - bins[0]) / 254
        x = torch.tensor([0.0, 1.0, -1.0, 3.0, -3.0])
        encoded = twohot(x, bins)
        decoded = twohot_decode(encoded, bins)
        assert torch.allclose(decoded, x, atol=bin_width.item() + 1e-4)


class TestSymlogBins:
    def test_default_length(self):
        bins = symlog_bins(255)
        assert bins.shape == (255,)

    def test_sorted_ascending(self):
        bins = symlog_bins(255)
        assert (bins[1:] > bins[:-1]).all()

    def test_custom_length(self):
        bins = symlog_bins(n_bins=64)
        assert bins.shape == (64,)


class TestUnimix:
    def test_output_sums_to_one(self):
        logits = torch.randn(4, 32, 32)
        mixed = unimix(logits, eps=0.01)
        assert torch.allclose(mixed.sum(-1), torch.ones(4, 32), atol=1e-5)

    def test_minimum_mass(self):
        logits = torch.zeros(32, 32)
        logits[0, 0] = 1000.0
        mixed = unimix(logits, eps=0.01)
        expected_min = 0.01 / 32
        assert mixed.min().item() >= expected_min * 0.99

    def test_non_negative(self):
        logits = torch.randn(8, 32)
        mixed = unimix(logits, eps=0.01)
        assert (mixed >= 0).all()

    def test_shape_preserved(self):
        logits = torch.randn(2, 3, 32)
        mixed = unimix(logits, eps=0.01)
        assert mixed.shape == logits.shape
