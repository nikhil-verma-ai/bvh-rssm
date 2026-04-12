#!/usr/bin/env python
"""
BVH-RSSM 3-phase training entry point.

Pipeline
--------
Phase 0 — Random data collection:
    Collect env transitions with a uniform random policy to seed the replay
    buffer before any world-model gradient steps.

Phase 1 — World model pretraining:
    Trains encoder, decoder, RSSM, reward_head, continue_head with the
    DreamerV3-style world model loss (prediction + dynamics KL + repr KL).

Phase 2 — BVH head training:
    Freezes world model weights, trains tau_head and hazard_head with the
    validity / survival losses.

Phase 3 — Joint fine-tuning + actor-critic (optional):
    Calls Trainer.train_phase3() if implemented; warns and skips otherwise.

Usage
-----
    python scripts/train.py           # full training run
    python scripts/train.py --fast    # fast mode (small model, 500 random steps)

Device priority: MPS (Apple Silicon) > CUDA > CPU (auto-detected).

Fast-mode dimensions match ShiftPendulum's actual observation / action space:
    obs_dim=3, action_dim=1 (Box -2..2, shape (1,))

Full-mode uses the same dims — ShiftPendulum is the training environment
because it has zero optional dependencies (pure Gymnasium + NumPy).
"""
from __future__ import annotations

import sys
import os
import warnings

# Ensure project root is importable when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from bvh_rssm.networks import RSSM, Encoder, Decoder, RewardHead, ContinueHead
from bvh_rssm.networks.heads import ValidityHead, HazardHead
from bvh_rssm.networks.actor_critic import Actor, Critic
from bvh_rssm.training.replay_buffer import ReplayBuffer
from bvh_rssm.training.trainer import Trainer, TrainerConfig
from bvh_rssm.training.collector import Collector
from bvh_rssm.training.experiment import set_seed, init_wandb


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    fast_mode: bool = False,
    obs_dim: int = 3,
    action_dim: int = 1,
) -> dict:
    """Construct all BVH-RSSM network components and return as a named dict.

    Architecture dimensions scale with fast_mode for quick local iteration.
    All modules are uninitialised (random weights) — training populates them.

    Args:
        fast_mode: If True, use small hidden dimensions suitable for CPU smoke tests.
        obs_dim:   Raw observation dimensionality (encoder input).  Default=3 for
                   ShiftPendulum (cos θ, sin θ, dθ/dt).
        action_dim: Action space dimensionality.  Default=1 for ShiftPendulum (Box(1,)).

    Returns:
        Dict mapping component names to nn.Module instances.  Keys:
          encoder, decoder, rssm, reward_head, continue_head,
          tau_head, hazard_head, actor, critic.
    """
    h_dim          = 128  if fast_mode else 512
    z_cats         = 8    if fast_mode else 32
    z_classes      = 8    if fast_mode else 32
    embed_dim      = 256  if fast_mode else 1024
    hidden_dim     = 128  if fast_mode else 512
    n_bins         = 32   if fast_mode else 255
    hazard_intervals = 8  if fast_mode else 16

    z_dim      = z_cats * z_classes
    latent_dim = h_dim + z_dim

    return {
        "encoder": Encoder(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=1 if fast_mode else 2,
        ),
        "decoder": Decoder(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            n_layers=1 if fast_mode else 2,
        ),
        "rssm": RSSM(
            h_dim=h_dim,
            z_cats=z_cats,
            z_classes=z_classes,
            obs_dim=embed_dim,
            action_dim=action_dim,
        ),
        "reward_head": RewardHead(
            latent_dim=latent_dim,
            n_bins=n_bins,
            hidden_dim=hidden_dim,
        ),
        "continue_head": ContinueHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ),
        "tau_head": ValidityHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
            n_bins=n_bins,
            hidden_dim=hidden_dim,
        ),
        "hazard_head": HazardHead(
            latent_dim=latent_dim,
            n_intervals=hazard_intervals,
            hidden_dim=hidden_dim,
        ),
        "actor": Actor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            discrete=False,   # ShiftPendulum uses Box action space
            hidden_dim=hidden_dim,
        ),
        "critic": Critic(
            latent_dim=latent_dim,
            n_bins=n_bins,
            hidden_dim=hidden_dim,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse CLI flags, run 3-phase training pipeline."""
    fast_mode = "--fast" in sys.argv
    seed      = 42

    # Device priority: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device_str = "mps"
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"

    print(f"[train] device={device_str}  fast_mode={fast_mode}  seed={seed}")

    set_seed(seed)

    # ------------------------------------------------------------------
    # Dimensions: ShiftPendulum — obs=(cos θ, sin θ, dθ/dt), action=Box(1,)
    # ------------------------------------------------------------------
    obs_dim    = 3
    action_dim = 1

    # Phase-step counts
    if fast_mode:
        random_steps  = 500
        phase1_steps  = 100
        phase2_steps  = 50
        phase3_steps  = 0
        buf_capacity  = 10_000
    else:
        random_steps  = 10_000
        phase1_steps  = 100_000
        phase2_steps  = 50_000
        phase3_steps  = 20_000
        buf_capacity  = 1_000_000

    # ------------------------------------------------------------------
    # Build model and replay buffer
    # ------------------------------------------------------------------
    model = build_model(fast_mode=fast_mode, obs_dim=obs_dim, action_dim=action_dim)

    buf = ReplayBuffer(
        capacity=buf_capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
        seq_len=16,
    )

    # ------------------------------------------------------------------
    # Move all model components to device before collection or training
    # ------------------------------------------------------------------
    device = torch.device(device_str)
    for m in model.values():
        if hasattr(m, "to"):
            m.to(device)

    # ------------------------------------------------------------------
    # Phase 0 — random data collection
    # ------------------------------------------------------------------
    print(f"[Phase 0] Collecting {random_steps} random steps …")
    collector = Collector(
        env_name="ShiftPendulum",
        model=model,
        replay_buffer=buf,
        device=device,
        fast_mode=fast_mode,
    )
    collected = collector.collect_steps(random_steps, random_policy=True)
    print(f"[Phase 0] Done — {collected} steps in buffer ({len(buf)} total)")

    # ------------------------------------------------------------------
    # Trainer config
    # ------------------------------------------------------------------
    run_tag = "fast" if fast_mode else "full"
    cfg = TrainerConfig(
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
        learning_rate=3e-4,
        grad_clip=100.0,
        batch_size=16,
        seq_len=16,
        log_every=10 if fast_mode else 1000,
        checkpoint_every=0,          # disable checkpoint saves during fast run
        device=device_str,
        seed=seed,
        run_dir=f"runs/{run_tag}/seed{seed}",
    )

    trainer = Trainer(model, buf, cfg)

    # ------------------------------------------------------------------
    # Phase 1 — world model pretraining
    # ------------------------------------------------------------------
    print(f"[Phase 1] World model pretraining for {phase1_steps} steps …")
    trainer.train_phase1()
    print("[Phase 1] Done")

    # ------------------------------------------------------------------
    # Phase 2 — BVH head training
    # ------------------------------------------------------------------
    print(f"[Phase 2] BVH head training for {phase2_steps} steps …")
    trainer.train_phase2()
    print("[Phase 2] Done")

    # ------------------------------------------------------------------
    # Phase 3 — joint fine-tuning + actor-critic (optional)
    # ------------------------------------------------------------------
    if phase3_steps > 0:
        if hasattr(trainer, "train_phase3"):
            print(f"[Phase 3] Joint fine-tuning for {phase3_steps} steps …")
            trainer.train_phase3()
            print("[Phase 3] Done")
        else:
            warnings.warn(
                f"phase3_steps={phase3_steps} requested but Trainer.train_phase3() "
                "is not yet implemented — skipping Phase 3.",
                stacklevel=1,
            )
            print("[Phase 3] Skipped (train_phase3 not yet implemented)")
    else:
        print("[Phase 3] Skipped (phase3_steps=0)")

    print("[train] All phases complete.")


if __name__ == "__main__":
    main()
