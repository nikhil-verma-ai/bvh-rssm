import pytest
import torch
import torch.nn as nn
from bvh_rssm.networks.common import MLP, LayerNormMLP


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(in_dim=16, out_dim=8, hidden_dim=32, n_layers=2)
        x = torch.randn(4, 16)
        out = mlp(x)
        assert out.shape == (4, 8)

    def test_gradient_flows(self):
        mlp = MLP(in_dim=8, out_dim=4, hidden_dim=16, n_layers=2)
        x = torch.randn(2, 8, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None

    def test_zero_hidden_layers(self):
        mlp = MLP(in_dim=8, out_dim=4, hidden_dim=16, n_layers=0)
        x = torch.randn(3, 8)
        out = mlp(x)
        assert out.shape == (3, 4)

    def test_batch_dims_preserved(self):
        mlp = MLP(in_dim=4, out_dim=2, hidden_dim=8, n_layers=1)
        x = torch.randn(2, 3, 4)
        out = mlp(x)
        assert out.shape == (2, 3, 2)

    def test_no_activation_on_output(self):
        """Output layer must have no activation (raw logits for heads)."""
        mlp = MLP(in_dim=4, out_dim=2, hidden_dim=8, n_layers=1)
        # Very negative input should produce finite (not zero-clamped) output
        x = torch.full((1, 4), -100.0)
        out = mlp(x)
        assert out.abs().sum() > 0  # not all-zero from ReLU clamp


class TestLayerNormMLP:
    def test_output_shape(self):
        mlp = LayerNormMLP(in_dim=16, out_dim=8, hidden_dim=32, n_layers=2)
        x = torch.randn(4, 16)
        out = mlp(x)
        assert out.shape == (4, 8)

    def test_gradient_flows(self):
        mlp = LayerNormMLP(in_dim=8, out_dim=4, hidden_dim=16, n_layers=2)
        x = torch.randn(2, 8, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None
