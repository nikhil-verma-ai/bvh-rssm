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
        """Output layer must be bare Linear — no activation."""
        mlp = MLP(in_dim=4, out_dim=2, hidden_dim=8, n_layers=1)
        last_module = list(mlp.net.children())[-1]
        assert isinstance(last_module, nn.Linear), (
            f"Last module should be Linear, got {type(last_module).__name__}"
        )


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

    def test_layer_order_is_linear_layernorm_silu(self):
        """Hidden blocks must be Linear → LayerNorm → SiLU in that order."""
        mlp = LayerNormMLP(in_dim=8, out_dim=4, hidden_dim=16, n_layers=2)
        children = list(mlp.net.children())
        # With n_layers=2: [Linear, LN, SiLU, Linear, LN, SiLU, Linear]
        # Verify first block ordering
        assert isinstance(children[0], nn.Linear)
        assert isinstance(children[1], nn.LayerNorm)
        assert isinstance(children[2], nn.SiLU)
        # Verify output layer has no activation
        assert isinstance(children[-1], nn.Linear)

    def test_batch_dims_preserved(self):
        mlp = LayerNormMLP(in_dim=4, out_dim=2, hidden_dim=8, n_layers=1)
        x = torch.randn(2, 3, 4)
        out = mlp(x)
        assert out.shape == (2, 3, 2)
