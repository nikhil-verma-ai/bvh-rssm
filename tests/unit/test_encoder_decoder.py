import pytest
import torch
from bvh_rssm.networks.encoder import Encoder
from bvh_rssm.networks.decoder import Decoder
from bvh_rssm.networks.rssm import RSSM, State


class TestEncoder:
    def setup_method(self):
        self.encoder = Encoder(obs_dim=16, embed_dim=32)

    def test_output_shape(self):
        obs = torch.randn(4, 16)
        embed = self.encoder(obs)
        assert embed.shape == (4, 32)

    def test_gradient_flows(self):
        obs = torch.randn(2, 16, requires_grad=True)
        embed = self.encoder(obs)
        embed.sum().backward()
        assert obs.grad is not None

    def test_symlog_applied_to_input(self):
        """Encoder applies symlog to raw observations before MLP."""
        obs_large = torch.full((1, 16), 1000.0)
        obs_small = torch.full((1, 16), 1.0)
        embed_large = self.encoder(obs_large)
        embed_small = self.encoder(obs_small)
        assert torch.isfinite(embed_large).all()
        assert torch.isfinite(embed_small).all()

    def test_batch_dims_preserved(self):
        obs = torch.randn(3, 4, 16)
        embed = self.encoder(obs)
        assert embed.shape == (3, 4, 32)


class TestDecoder:
    def setup_method(self):
        self.h_dim = 32
        self.z_dim = 16
        self.obs_dim = 8
        self.decoder = Decoder(
            latent_dim=self.h_dim + self.z_dim,
            obs_dim=self.obs_dim,
        )

    def test_output_shape(self):
        latent = torch.randn(4, self.h_dim + self.z_dim)
        mean, log_std = self.decoder(latent)
        assert mean.shape == (4, self.obs_dim)

    def test_gradient_flows(self):
        latent = torch.randn(2, self.h_dim + self.z_dim, requires_grad=True)
        mean, log_std = self.decoder(latent)
        mean.sum().backward()
        assert latent.grad is not None

    def test_nll_loss_computation(self):
        """Decoder NLL loss must be finite and differentiable (computed in symlog space)."""
        from bvh_rssm.utils import symlog
        latent = torch.randn(4, self.h_dim + self.z_dim)
        # Targets must be symlog-transformed for loss computation
        obs_target = torch.randn(4, self.obs_dim)
        mean_symlog, log_std = self.decoder.decode_symlog(latent)
        std = log_std.exp()
        # NLL in symlog space (DreamerV3 canonical)
        nll = 0.5 * ((symlog(obs_target) - mean_symlog) / std).pow(2) + log_std + 0.9189
        loss = nll.mean()
        assert torch.isfinite(loss)
        self.decoder.zero_grad()
        loss.backward()
        assert any(p.grad is not None for p in self.decoder.parameters())

    def test_symexp_applied_to_output(self):
        """Decoder applies symexp to mean output to recover original scale."""
        latent = torch.randn(1, self.h_dim + self.z_dim)
        mean, _ = self.decoder(latent)
        assert torch.isfinite(mean).all()


class TestEncoderDecoderWithRSSM:
    def test_encoder_output_feeds_rssm_observe(self):
        obs_dim = 8
        embed_dim = 32
        h_dim = 16
        z_cats, z_classes = 2, 4

        encoder = Encoder(obs_dim=obs_dim, embed_dim=embed_dim)
        rssm = RSSM(h_dim=h_dim, z_cats=z_cats, z_classes=z_classes,
                    obs_dim=embed_dim, action_dim=2)

        obs = torch.randn(3, obs_dim)
        action = torch.randn(3, 2)
        state = rssm.initial_state(3)

        embed = encoder(obs)
        assert embed.shape == (3, embed_dim)

        posterior_logits, next_state = rssm.observe(embed, action, state)
        assert next_state.h.shape == (3, h_dim)

    def test_decoder_takes_rssm_latent(self):
        h_dim, z_dim, obs_dim = 16, 8, 4
        decoder = Decoder(latent_dim=h_dim + z_dim, obs_dim=obs_dim)
        h = torch.randn(2, h_dim)
        z = torch.randn(2, z_dim)
        latent = torch.cat([h, z], dim=-1)
        mean, log_std = decoder(latent)
        assert mean.shape == (2, obs_dim)
        assert log_std.shape == (2, obs_dim)
