#!/usr/bin/env python3
"""
Phase 3 BVH-Gated Policy experiment.

Part A (--mode eval):
    Loads trained Phase 1+2 checkpoint. Runs N episodes under two policies:
      BVH policy:     STALE → zero-action fallback; HIGH/DIM → actor
      Vanilla policy: always actor, ignores RouterState
    Reports delta_return = E[BVH] - E[Vanilla].

Part B (--mode train):
    Trains actor-critic from the checkpoint using train_phase3().
    --gated enables imagination gating (BVH actor).
    Without --gated: vanilla actor (fixed K=16).

Usage
-----
    # Part A
    python3 scripts/experiment_phase3.py \\
        --mode eval \\
        --checkpoint checkpoints/sd_p1_v2.pt \\
        --n-episodes 200 \\
        --out results/phase3_eval_report.json

    # Part B (BVH actor)
    python3 scripts/experiment_phase3.py \\
        --mode train \\
        --checkpoint checkpoints/sd_p1_v2.pt \\
        --phase3-steps 50000 \\
        --gated \\
        --out results/phase3_train_report.json

    # Part B (vanilla baseline)
    python3 scripts/experiment_phase3.py \\
        --mode train \\
        --checkpoint checkpoints/sd_p1_v2.pt \\
        --phase3-steps 50000 \\
        --out results/phase3_vanilla_report.json
"""
from __future__ import annotations

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from bvh_rssm.networks import RSSM, Encoder, Decoder, RewardHead, ContinueHead
from bvh_rssm.networks.heads import ValidityHead, HazardHead
from bvh_rssm.networks.actor_critic import Actor, Critic
from bvh_rssm.causal.router import AdaptivePolicyRouter, RouterState
from bvh_rssm.training.replay_buffer import ReplayBuffer
from bvh_rssm.training.trainer import Trainer, TrainerConfig
from bvh_rssm.training.losses import validity_loss, survival_loss
from bvh_rssm.training.experiment import set_seed
from bvh_rssm.envs.sensor_drift import SensorDrift

# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match sd_p1_v2.pt architecture (h_dim=512)
# ──────────────────────────────────────────────────────────────────────────────
K          = 16
DRIFT_RATE = 0.5 / K    # ≈ 0.03125
OBS_DIM    = 17
ACTION_DIM = 6
H_DIM      = 512         # must match checkpoint
Z_CATS     = 8
Z_CLASSES  = 8
EMBED_DIM  = 256
HIDDEN_DIM = 256
N_BINS     = 64

WM_KEYS   = ["encoder", "decoder", "rssm", "reward_head", "continue_head"]
HEAD_KEYS = ["tau_head", "hazard_head"]

P2_HEAD_LR  = 1e-3
P2_SEQ_LEN  = 64
P2_BURNIN   = 32
P2_LOG_EVERY = 5_000


# ──────────────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> dict:
    z_dim      = Z_CATS * Z_CLASSES
    latent_dim = H_DIM + z_dim

    model = {
        "encoder":       Encoder(OBS_DIM, EMBED_DIM, hidden_dim=HIDDEN_DIM, n_layers=2),
        "decoder":       Decoder(latent_dim, OBS_DIM, hidden_dim=HIDDEN_DIM, n_layers=2),
        "rssm":          RSSM(H_DIM, Z_CATS, Z_CLASSES, EMBED_DIM, ACTION_DIM),
        "reward_head":   RewardHead(latent_dim, N_BINS, hidden_dim=HIDDEN_DIM),
        "continue_head": ContinueHead(latent_dim, hidden_dim=HIDDEN_DIM),
        "tau_head":      ValidityHead(latent_dim, ACTION_DIM, n_bins=N_BINS,
                                     hidden_dim=HIDDEN_DIM, max_horizon=K + 5),
        "hazard_head":   HazardHead(latent_dim, n_intervals=K, hidden_dim=HIDDEN_DIM),
        "actor":         Actor(latent_dim, ACTION_DIM, discrete=False, hidden_dim=HIDDEN_DIM),
        "critic":        Critic(latent_dim, n_bins=N_BINS, hidden_dim=HIDDEN_DIM),
    }
    for m in model.values():
        m.to(device)
    return model


def load_checkpoint(model: dict, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt.pop("_meta", None)
    loaded = []
    for k in list(WM_KEYS) + ["tau_head", "hazard_head"]:
        if k in ckpt:
            model[k].load_state_dict(ckpt[k], strict=False)
            loaded.append(k)
    print(f"  Loaded: {loaded}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: retrain BVH heads from loaded WM checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def phase2_train_heads(
    model: dict,
    buf: ReplayBuffer,
    device: torch.device,
    n_steps: int,
) -> None:
    """Train tau_head and hazard_head with frozen world model.

    Mirrors experiment_sensordrift.py phase2_train() logic.
    Modifies model in-place.
    """
    # Freeze WM, unfreeze heads
    for k in WM_KEYS:
        for p in model[k].parameters():
            p.requires_grad_(False)
    for k in HEAD_KEYS:
        for p in model[k].parameters():
            p.requires_grad_(True)

    for m in model.values():
        m.train()

    head_params = [p for k in HEAD_KEYS for p in model[k].parameters()]
    opt = torch.optim.Adam(head_params, lr=P2_HEAD_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-5)

    rssm   = model["rssm"]
    enc    = model["encoder"]
    tau_h  = model["tau_head"]
    haz_h  = model["hazard_head"]

    # Seed buffer with more SensorDrift experience for Phase 2 (longer sequences)
    env     = SensorDrift(drift_rate=DRIFT_RATE, seed=99)
    obs_np, _ = env.reset(seed=99)
    print("[p2] Collecting 5000 seed steps …")
    for _ in range(5000):
        act = env.action_space.sample().astype(np.float32)
        obs_next, rew, term, trunc, info = env.step(act)
        buf.push(
            obs=obs_np, action=act, reward=float(rew),
            terminated=bool(term or trunc),
            oracle_tau=float(info.get("oracle_tau", K)),
            is_interventionist=False, rng_state={},
        )
        obs_np = obs_next if not (term or trunc) else env.reset()[0]
    env.close()

    t0 = time.time()
    for step in range(n_steps):
        batch      = buf.sample(16, P2_SEQ_LEN)
        obs        = batch["obs"].to(device)
        actions    = batch["action"].to(device)
        oracle_tau = batch["oracle_tau"].float().to(device)

        with torch.no_grad():
            state   = rssm.initial_state(16, device=device)
            latents = []
            for t in range(obs.shape[1]):
                emb = enc(obs[:, t])
                _, state = rssm.observe(emb, actions[:, t], state)
                latents.append(rssm.get_latent(state))
            latents_post = latents[P2_BURNIN:]

        post_actions = actions[:, P2_BURNIN:]
        post_tau     = oracle_tau[:, P2_BURNIN:]
        flat_lat     = torch.stack(latents_post, 1).reshape(-1, latents_post[0].shape[-1])
        flat_act     = post_actions.reshape(-1, post_actions.shape[-1])
        flat_tau     = post_tau.reshape(-1)
        event_t      = flat_tau.long().clamp(0, K - 1)
        ev_occ       = (flat_tau < K)

        opt.zero_grad()
        v_loss = validity_loss(tau_h, flat_lat, flat_act, flat_tau, stop_grad=True)
        s_loss = survival_loss(haz_h, flat_lat, event_t, ev_occ, use_all_sources=False)
        (v_loss + s_loss).backward()
        torch.nn.utils.clip_grad_norm_(head_params, 5.0)
        opt.step()
        scheduler.step()

        if step % P2_LOG_EVERY == 0:
            print(
                f"  P2 step {step:6d}/{n_steps}  "
                f"tau_loss={v_loss.item():.3f}  "
                f"surv_loss={s_loss.item():.3f}  "
                f"({time.time()-t0:.0f}s)"
            )

    print(f"[p2] Done — {n_steps} steps in {time.time()-t0:.0f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Episode rollout
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_episode(
    model: dict,
    seed: int,
    device: torch.device,
    use_bvh_gate: bool,
    stale_threshold: float = 3.0,
    max_steps: int = 500,
) -> dict:
    """Run one episode and return metrics.

    BVH gating uses a direct tau_hat threshold: when tau_hat < stale_threshold,
    the model is within `stale_threshold` steps of going stale and zero-action
    is the safe fallback. This is more robust than the router's survival-curve
    thresholds because the tau_head and hazard_head co-track tau* when both are
    well-trained — making the adaptive thresholds move together and preventing
    STALE from ever firing via the router path alone.

    Args:
        stale_threshold: Use zero-action when tau_hat < this value (default 3.0).
            On SensorDrift (K=16), this fires in the last ~3/16 ≈ 19% of each
            drift cycle — right when imagined dynamics are maximally noisy.

    Returns:
        dict with: total_return, stale_triggered, first_stale_step, n_steps
    """
    env       = SensorDrift(drift_rate=DRIFT_RATE, seed=seed)
    obs_np, _ = env.reset(seed=seed)

    rssm  = model["rssm"]
    enc   = model["encoder"]
    actor = model["actor"]
    tau_h = model["tau_head"]

    state     = rssm.initial_state(1, device=device)
    prev_act  = torch.zeros(1, ACTION_DIM, device=device)
    total_ret = 0.0
    stale_triggered  = False
    first_stale_step = -1

    for t in range(max_steps):
        obs_t = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0).to(device)

        # Update RSSM state
        emb = enc(obs_t)
        _, state = rssm.observe(emb, prev_act, state)
        lat = rssm.get_latent(state)  # [1, latent_dim]

        # Compute τ̂ from trained validity head
        tau_hat = float(tau_h.decode(tau_h(lat, prev_act, stop_grad=True)).mean().item())

        # Policy decision — BVH gate: zero-action when τ̂ below absolute threshold
        if use_bvh_gate and tau_hat < stale_threshold:
            action_np = np.zeros(ACTION_DIM, dtype=np.float32)
            if not stale_triggered:
                stale_triggered  = True
                first_stale_step = t
        else:
            actor_out = actor(lat)
            mean, log_std = actor_out
            std = log_std.exp()
            act = mean + std * torch.randn_like(mean)
            action_np = act.squeeze(0).cpu().numpy().astype(np.float32)

        obs_np, reward, term, trunc, _ = env.step(action_np)
        total_ret += float(reward)

        prev_act = torch.tensor(action_np, device=device).unsqueeze(0)

        if term or trunc:
            break

    env.close()
    return {
        "total_return":     total_ret,
        "stale_triggered":  stale_triggered,
        "first_stale_step": first_stale_step,
        "n_steps":          t + 1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Part A: Inference-only evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(args) -> dict:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[eval] device={device}  episodes={args.n_episodes}")

    model = build_model(device)
    if args.checkpoint:
        print(f"[eval] Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint)

    if args.p2_steps > 0:
        print(f"[eval] Running Phase 2 head training ({args.p2_steps} steps) …")
        buf_p2 = ReplayBuffer(capacity=50_000, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        phase2_train_heads(model, buf_p2, device, args.p2_steps)

    for m in model.values():
        m.eval()

    seeds = list(range(args.n_episodes))  # fixed seeds 0..N-1

    stale_threshold = args.stale_threshold
    print(f"[eval] BVH stale_threshold={stale_threshold}")

    print("[eval] Running BVH policy …")
    bvh_results = []
    for i, seed in enumerate(seeds):
        r = run_episode(model, seed=seed, device=device, use_bvh_gate=True,
                        stale_threshold=stale_threshold)
        bvh_results.append(r)
        if (i + 1) % 20 == 0:
            mean_ret = np.mean([x["total_return"] for x in bvh_results])
            print(f"  BVH {i+1}/{args.n_episodes}  mean_return={mean_ret:.2f}")

    print("[eval] Running Vanilla policy …")
    vanilla_results = []
    for i, seed in enumerate(seeds):
        r = run_episode(model, seed=seed, device=device, use_bvh_gate=False)
        vanilla_results.append(r)
        if (i + 1) % 20 == 0:
            mean_ret = np.mean([x["total_return"] for x in vanilla_results])
            print(f"  Vanilla {i+1}/{args.n_episodes}  mean_return={mean_ret:.2f}")

    bvh_rets     = [r["total_return"] for r in bvh_results]
    vanilla_rets = [r["total_return"] for r in vanilla_results]

    fail_thresh  = -50.0
    bvh_fail     = sum(1 for r in bvh_rets if r < fail_thresh) / len(bvh_rets)
    vanilla_fail = sum(1 for r in vanilla_rets if r < fail_thresh) / len(vanilla_rets)

    stale_eps       = [r for r in bvh_results if r["stale_triggered"]]
    stale_rate      = len(stale_eps) / args.n_episodes
    mean_stale_step = (
        float(np.mean([r["first_stale_step"] for r in stale_eps]))
        if stale_eps else -1.0
    )

    report = {
        "mode":                    "eval",
        "n_episodes":              args.n_episodes,
        "checkpoint":              args.checkpoint,
        "stale_threshold":         stale_threshold,
        "bvh_mean_return":         float(np.mean(bvh_rets)),
        "vanilla_mean_return":     float(np.mean(vanilla_rets)),
        "delta_return_inference":  float(np.mean(bvh_rets) - np.mean(vanilla_rets)),
        "failure_rate_bvh":        float(bvh_fail),
        "failure_rate_vanilla":    float(vanilla_fail),
        "stale_trigger_rate":      float(stale_rate),
        "mean_stale_trigger_step": float(mean_stale_step),
    }

    print("\n[eval] Results:")
    print(f"  BVH mean return:      {report['bvh_mean_return']:.2f}")
    print(f"  Vanilla mean return:  {report['vanilla_mean_return']:.2f}")
    print(f"  delta_return:         {report['delta_return_inference']:+.2f}")
    print(f"  failure_rate BVH:     {report['failure_rate_bvh']:.3f}")
    print(f"  failure_rate Vanilla: {report['failure_rate_vanilla']:.3f}")
    print(f"  stale_trigger_rate:   {report['stale_trigger_rate']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[eval] Report saved → {args.out}")
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Part B: Training + eval
# ──────────────────────────────────────────────────────────────────────────────

def run_train(args) -> dict:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[train] device={device}  steps={args.phase3_steps}  gated={args.gated}")

    model = build_model(device)
    if args.checkpoint:
        print(f"[train] Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint)

    # Seed replay buffer — used for both Phase 2 (if requested) and Phase 3
    buf = ReplayBuffer(capacity=100_000, obs_dim=OBS_DIM, action_dim=ACTION_DIM)

    if args.p2_steps > 0:
        print(f"[train] Running Phase 2 head training ({args.p2_steps} steps) …")
        phase2_train_heads(model, buf, device, args.p2_steps)
    env     = SensorDrift(drift_rate=DRIFT_RATE, seed=42)
    obs_np, _ = env.reset(seed=42)
    print("[train] Seeding replay buffer (2000 random steps) …")
    for _ in range(2000):
        act = env.action_space.sample().astype(np.float32)
        obs_next, rew, term, trunc, info = env.step(act)
        buf.push(
            obs=obs_np,
            action=act,
            reward=float(rew),
            terminated=bool(term or trunc),
            oracle_tau=float(info.get("oracle_tau", K)),
            is_interventionist=False,
            rng_state={},
        )
        if term or trunc:
            obs_np, _ = env.reset()
        else:
            obs_np = obs_next
    env.close()

    cfg = TrainerConfig(
        phase1_steps=0,
        phase2_steps=0,
        phase3_steps=args.phase3_steps,
        learning_rate=1e-4,
        grad_clip=100.0,
        batch_size=16,
        seq_len=16,
        gamma=0.99,
        lambda_=0.95,
        entropy_coef=3e-4,
        cf_margin=3.0,
        lambda_cf=0.1,
        log_every=500,
        checkpoint_every=10_000,
        device=str(device),
        seed=42,
        run_dir=f"runs/phase3_{'gated' if args.gated else 'vanilla'}",
    )

    trainer = Trainer(model, buf, cfg)
    print(f"[train] Phase 3 training: {args.phase3_steps} steps …")
    t0 = time.time()
    trainer.train_phase3(imagination_gating=args.gated)
    elapsed = time.time() - t0
    print(f"[train] Training complete in {elapsed:.0f}s")

    # Evaluate trained actor
    for m in model.values():
        m.eval()
    seeds = list(range(200))

    print("[train] Evaluating trained actor (200 episodes) …")
    trained_results = []
    for i, seed in enumerate(seeds):
        r = run_episode(model, seed=seed, device=device, use_bvh_gate=args.gated,
                        stale_threshold=args.stale_threshold)
        trained_results.append(r)
        if (i + 1) % 20 == 0:
            mean_ret = np.mean([x["total_return"] for x in trained_results])
            print(f"  {i+1}/200  mean_return={mean_ret:.2f}")

    rets        = [r["total_return"] for r in trained_results]
    fail_thresh = -50.0
    fail_rate   = sum(1 for r in rets if r < fail_thresh) / len(rets)

    stale_eps       = [r for r in trained_results if r["stale_triggered"]]
    stale_rate      = len(stale_eps) / len(trained_results)
    mean_stale_step = (
        float(np.mean([r["first_stale_step"] for r in stale_eps]))
        if stale_eps else -1.0
    )

    report = {
        "mode":                    "train",
        "gated":                   args.gated,
        "n_episodes":              200,
        "phase3_steps":            args.phase3_steps,
        "checkpoint":              args.checkpoint,
        "trained_mean_return":     float(np.mean(rets)),
        "trained_std_return":      float(np.std(rets)),
        "failure_rate":            float(fail_rate),
        "stale_trigger_rate":      float(stale_rate),
        "mean_stale_trigger_step": float(mean_stale_step),
        "elapsed_s":               float(elapsed),
    }

    label = "BVH-gated" if args.gated else "Vanilla"
    print(f"\n[train] Results ({label} actor):")
    print(f"  mean_return:         {report['trained_mean_return']:.2f}")
    print(f"  failure_rate:        {report['failure_rate']:.3f}")
    print(f"  stale_trigger_rate:  {report['stale_trigger_rate']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[train] Report saved → {args.out}")
    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 BVH-Gated Policy experiment")
    p.add_argument("--mode", choices=["eval", "train"], required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to Phase 1+2 checkpoint (e.g. checkpoints/sd_p1_v2.pt)")
    p.add_argument("--n-episodes", type=int, default=200,
                   help="Number of eval episodes (Part A, default: 200)")
    p.add_argument("--p2-steps", type=int, default=50_000,
                   help="Phase 2 head training steps before eval/train (default: 50000). "
                        "Set 0 to skip if checkpoint already contains tau_head weights.")
    p.add_argument("--phase3-steps", type=int, default=50_000,
                   help="Phase 3 training steps (Part B, default: 50000)")
    p.add_argument("--gated", action="store_true",
                   help="Enable BVH imagination gating during training (Part B)")
    p.add_argument("--stale-threshold", type=float, default=3.0,
                   help="Use zero-action when tau_hat < this value (default: 3.0). "
                        "On SensorDrift (K=16), fires in the last ~3 steps of each "
                        "16-step drift cycle when imagined dynamics are most noisy.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/phase3_report.json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    set_seed(args.seed)
    if args.mode == "eval":
        run_eval(args)
    else:
        run_train(args)
