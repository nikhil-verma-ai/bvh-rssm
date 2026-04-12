"""100-step smoke train: validates that phases 1-2 run without crash."""
import pytest
import torch
import numpy as np
from bvh_rssm.networks import RSSM, Encoder, Decoder, RewardHead, ContinueHead
from bvh_rssm.networks.heads import ValidityHead, HazardHead
from bvh_rssm.training.replay_buffer import ReplayBuffer
from bvh_rssm.training.losses import world_model_loss, validity_loss
from bvh_rssm.training.experiment import set_seed
from bvh_rssm.utils.rng import save_rng_state


@pytest.fixture
def fast_model():
    obs_dim = 8
    action_dim = 3
    h_dim = 32
    z_cats, z_classes = 4, 4
    embed_dim = 16
    z_dim = z_cats * z_classes
    latent_dim = h_dim + z_dim

    encoder = Encoder(obs_dim=obs_dim, embed_dim=embed_dim, hidden_dim=64, n_layers=1)
    decoder = Decoder(latent_dim=latent_dim, obs_dim=obs_dim, hidden_dim=64, n_layers=1)
    rssm = RSSM(h_dim=h_dim, z_cats=z_cats, z_classes=z_classes,
                obs_dim=embed_dim, action_dim=action_dim)
    reward_head = RewardHead(latent_dim=latent_dim, n_bins=32, hidden_dim=64)
    continue_head = ContinueHead(latent_dim=latent_dim, hidden_dim=64)
    tau_head = ValidityHead(latent_dim=latent_dim, action_dim=action_dim,
                            n_bins=32, hidden_dim=64)
    hazard_head = HazardHead(latent_dim=latent_dim, n_intervals=8, hidden_dim=64)

    return {
        "obs_dim": obs_dim, "action_dim": action_dim, "latent_dim": latent_dim,
        "encoder": encoder, "decoder": decoder, "rssm": rssm,
        "reward_head": reward_head, "continue_head": continue_head,
        "tau_head": tau_head, "hazard_head": hazard_head,
    }


def test_phase1_smoke(fast_model):
    """10 steps of phase 1 training must not crash and produce finite loss."""
    set_seed(0)
    m = fast_model
    obs_dim, action_dim = m["obs_dim"], m["action_dim"]
    B, T = 4, 8

    params = (list(m["encoder"].parameters()) + list(m["decoder"].parameters()) +
              list(m["rssm"].parameters()) + list(m["reward_head"].parameters()) +
              list(m["continue_head"].parameters()))
    optimizer = torch.optim.Adam(params, lr=1e-3)

    buf = ReplayBuffer(capacity=1000, obs_dim=obs_dim, action_dim=action_dim, seq_len=T)
    for _ in range(200):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        rng = save_rng_state()
        buf.push(obs, action, 0.0, False, 10, False, rng)

    losses = []
    for step in range(10):
        batch = buf.sample(B, T)
        optimizer.zero_grad()
        result = world_model_loss(
            batch["obs"], batch["action"], batch["reward"], torch.ones(B, T),
            m["encoder"], m["decoder"], m["rssm"],
            m["reward_head"], m["continue_head"],
        )
        result["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 100.0)
        optimizer.step()
        losses.append(result["total"].item())

    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert len(losses) == 10


def test_phase2_smoke(fast_model):
    """5 steps of phase 2: frozen world model, training tau head."""
    set_seed(0)
    m = fast_model
    action_dim, latent_dim = m["action_dim"], m["latent_dim"]
    B = 4

    tau_params = list(m["tau_head"].parameters())
    optimizer = torch.optim.Adam(tau_params, lr=1e-3)

    for _ in range(5):
        latent = torch.randn(B, latent_dim)
        action = torch.randn(B, action_dim)
        oracle_tau = torch.randint(1, 50, (B,)).float()

        optimizer.zero_grad()
        v_loss = validity_loss(m["tau_head"], latent, action, oracle_tau, stop_grad=True)
        v_loss.backward()
        optimizer.step()
        assert torch.isfinite(v_loss)
