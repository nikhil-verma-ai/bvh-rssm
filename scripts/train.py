#!/usr/bin/env python
"""
BVH-RSSM training entry point.

Usage:
    python scripts/train.py           # defaults
    python scripts/train.py --fast    # fast mode (Mac testing)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from bvh_rssm.networks import RSSM, Encoder, Decoder, RewardHead, ContinueHead
from bvh_rssm.networks.heads import ValidityHead, HazardHead
from bvh_rssm.training.replay_buffer import ReplayBuffer
from bvh_rssm.training.trainer import Trainer, TrainerConfig
from bvh_rssm.training.experiment import set_seed, init_wandb


def build_model(fast_mode: bool = False, action_dim: int = 3, obs_dim: int = 8):
    """Construct all network components and return as a named dict.

    Args:
        fast_mode: If True, use smaller dims for local iteration / CI.
        action_dim: Dimensionality of the continuous action space.
        obs_dim: Dimensionality of the observation vector.

    Returns:
        Dict mapping component names to nn.Module instances.
    """
    h_dim = 128 if fast_mode else 512
    z_cats = 8 if fast_mode else 32
    z_classes = 8 if fast_mode else 32
    embed_dim = 256 if fast_mode else 1024
    z_dim = z_cats * z_classes
    latent_dim = h_dim + z_dim
    n_bins = 32 if fast_mode else 255
    hazard_intervals = 8 if fast_mode else 16

    return {
        "encoder": Encoder(obs_dim=obs_dim, embed_dim=embed_dim),
        "decoder": Decoder(latent_dim=latent_dim, obs_dim=obs_dim),
        "rssm": RSSM(h_dim=h_dim, z_cats=z_cats, z_classes=z_classes,
                     obs_dim=embed_dim, action_dim=action_dim),
        "reward_head": RewardHead(latent_dim=latent_dim, n_bins=n_bins),
        "continue_head": ContinueHead(latent_dim=latent_dim),
        "tau_head": ValidityHead(latent_dim=latent_dim, action_dim=action_dim, n_bins=n_bins),
        "hazard_head": HazardHead(latent_dim=latent_dim, n_intervals=hazard_intervals),
    }


def main():
    fast_mode = "--fast" in sys.argv
    seed = 42
    # Device priority: MPS (Apple Silicon) > CUDA > CPU
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    set_seed(seed)

    model = build_model(fast_mode=fast_mode)

    buf = ReplayBuffer(
        capacity=10_000 if fast_mode else 1_000_000,
        obs_dim=8,
        action_dim=3,
        seq_len=16,
    )

    cfg = TrainerConfig(
        phase1_steps=100 if fast_mode else 100_000,
        phase2_steps=50 if fast_mode else 50_000,
        phase3_steps=0,
        device=device,
        seed=seed,
        run_dir=f"runs/{'fast' if fast_mode else 'full'}/seed{seed}",
    )

    trainer = Trainer(model, buf, cfg)
    print(f"Training on {device}, fast_mode={fast_mode}")
    print("Note: buf is empty — populate before training in real usage")


if __name__ == "__main__":
    main()
