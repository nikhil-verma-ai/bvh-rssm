import torch
import pytest
from bvh_rssm.training.losses import (
    world_model_loss,
    kl_loss,
    validity_loss,
    counterfactual_loss,
)
from bvh_rssm.networks import RSSM, Encoder, Decoder, RewardHead, ContinueHead
from bvh_rssm.networks.heads import ValidityHead


B, T, obs_dim, action_dim = 4, 8, 8, 3
h_dim, z_cats, z_classes = 32, 4, 4
embed_dim = 16
z_dim = z_cats * z_classes
latent_dim = h_dim + z_dim


def make_model():
    encoder = Encoder(obs_dim=obs_dim, embed_dim=embed_dim, hidden_dim=64, n_layers=1)
    decoder = Decoder(latent_dim=latent_dim, obs_dim=obs_dim, hidden_dim=64, n_layers=1)
    rssm = RSSM(h_dim=h_dim, z_cats=z_cats, z_classes=z_classes,
                obs_dim=embed_dim, action_dim=action_dim)
    reward_head = RewardHead(latent_dim=latent_dim, n_bins=32, hidden_dim=64)
    continue_head = ContinueHead(latent_dim=latent_dim, hidden_dim=64)
    return encoder, decoder, rssm, reward_head, continue_head


class TestKLLoss:
    def test_kl_loss_finite(self):
        posterior = torch.randn(B, z_cats, z_classes)
        prior = torch.randn(B, z_cats, z_classes)
        loss = kl_loss(posterior, prior)
        assert torch.isfinite(loss)

    def test_kl_loss_nonneg(self):
        posterior = torch.randn(B, z_cats, z_classes)
        prior = torch.randn(B, z_cats, z_classes)
        loss = kl_loss(posterior, prior)
        assert loss.item() >= 0.0

    def test_kl_loss_differentiable(self):
        posterior = torch.randn(B, z_cats, z_classes, requires_grad=True)
        prior = torch.randn(B, z_cats, z_classes)
        loss = kl_loss(posterior, prior)
        loss.backward()
        assert posterior.grad is not None


class TestWorldModelLoss:
    def test_world_model_loss_returns_dict(self):
        encoder, decoder, rssm, reward_head, continue_head = make_model()
        obs = torch.randn(B, T, obs_dim)
        actions = torch.randn(B, T, action_dim)
        rewards = torch.randn(B, T)
        continues = torch.ones(B, T)
        result = world_model_loss(
            obs, actions, rewards, continues,
            encoder, decoder, rssm, reward_head, continue_head,
        )
        assert "total" in result
        assert "prediction" in result
        assert "dynamics" in result
        assert "representation" in result

    def test_world_model_loss_finite(self):
        encoder, decoder, rssm, reward_head, continue_head = make_model()
        obs = torch.randn(B, T, obs_dim)
        actions = torch.randn(B, T, action_dim)
        rewards = torch.randn(B, T)
        continues = torch.ones(B, T)
        result = world_model_loss(
            obs, actions, rewards, continues,
            encoder, decoder, rssm, reward_head, continue_head,
        )
        assert torch.isfinite(result["total"])

    def test_world_model_loss_differentiable(self):
        encoder, decoder, rssm, reward_head, continue_head = make_model()
        obs = torch.randn(B, T, obs_dim, requires_grad=True)
        actions = torch.randn(B, T, action_dim)
        rewards = torch.randn(B, T)
        continues = torch.ones(B, T)
        result = world_model_loss(
            obs, actions, rewards, continues,
            encoder, decoder, rssm, reward_head, continue_head,
        )
        result["total"].backward()
        assert obs.grad is not None


class TestValidityLoss:
    def test_validity_loss_finite(self):
        tau_head = ValidityHead(latent_dim=latent_dim, action_dim=action_dim,
                                n_bins=32, hidden_dim=64)
        latent = torch.randn(B, latent_dim)
        action = torch.randn(B, action_dim)
        oracle_tau = torch.randint(0, 50, (B,)).float()
        loss = validity_loss(tau_head, latent, action, oracle_tau, stop_grad=True)
        assert torch.isfinite(loss)

    def test_validity_loss_stop_grad(self):
        tau_head = ValidityHead(latent_dim=latent_dim, action_dim=action_dim,
                                n_bins=32, hidden_dim=64)
        latent = torch.randn(B, latent_dim, requires_grad=True)
        action = torch.randn(B, action_dim)
        oracle_tau = torch.ones(B) * 10.0
        loss = validity_loss(tau_head, latent, action, oracle_tau, stop_grad=True)
        loss.backward()
        assert latent.grad is None or latent.grad.abs().sum() == 0


class TestCounterfactualLoss:
    def test_cf_loss_nonneg(self):
        tau_obs = torch.ones(B) * 20.0
        tau_int = torch.ones(B) * 10.0
        loss = counterfactual_loss(tau_int, tau_obs, margin=3.0)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_cf_loss_zero_when_intervention_helps(self):
        """When tau_int >= tau_obs + margin, loss is 0."""
        tau_obs = torch.ones(B) * 5.0
        tau_int = torch.ones(B) * 20.0
        loss = counterfactual_loss(tau_int, tau_obs, margin=3.0)
        assert loss.item() == 0.0
