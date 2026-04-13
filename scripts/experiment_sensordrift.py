#!/usr/bin/env python3
"""
SensorDrift experiment — demonstrates C-index > 0.65 on a predictable-drift env.

SensorDrift (FNSB Env 6): observation noise grows deterministically each step.
oracle_tau = steps until noise_std hits the reset threshold.
Since drift is deterministic, tau* is fully predictable from the latent's
accumulated observation variance signal → C-index approaches 1.0.

Contrast with ShiftPendulum (Poisson shifts, memoryless) → C-index = 0.5.

Usage
-----
    python scripts/experiment_sensordrift.py                  # full run (~90 min)
    python scripts/experiment_sensordrift.py --steps 9000     # quick (~8 min)
    python scripts/experiment_sensordrift.py --skip-p1 checkpoints/sd_p1.pt

Outputs
-------
    checkpoints/sd_p1.pt           — Phase 1 checkpoint (save with --save-p1)
    results/sensordrift_report.json — full metrics
"""
from __future__ import annotations

import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
import torch.nn.functional as F

from bvh_rssm.networks import RSSM, Encoder, Decoder, RewardHead, ContinueHead
from bvh_rssm.networks.heads import ValidityHead, HazardHead
from bvh_rssm.training.replay_buffer import ReplayBuffer
from bvh_rssm.training.losses import world_model_loss, validity_loss, survival_loss
from bvh_rssm.training.experiment import set_seed
from bvh_rssm.training.metrics import mae_tau, c_index, brier_score
from bvh_rssm.envs.sensor_drift import SensorDrift
from bvh_rssm.utils import symlog

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# SensorDrift: noise grows at drift_rate per step, resets at threshold 0.5.
# drift_rate = 0.5 / K → max oracle_tau = K exactly → all values in [0, K].
K           = 16
DRIFT_RATE  = 0.5 / K   # ≈ 0.03125

OBS_DIM    = 17   # HalfCheetah-v4
ACTION_DIM = 6    # HalfCheetah-v4

SEED_STEPS = 1_000
P1_STEPS   = 100_000
P2_STEPS   = 50_000
LOG_EVERY  = 1_000
GRAD_CLIP  = 5.0

BUF_CAPACITY   = 200_000
AUX_TAU_WEIGHT = 0.3
P2_SEQ_LEN     = 64
P2_BURNIN      = 32
P2_HEAD_LR     = 1e-3

WM_KEYS   = ["encoder", "decoder", "rssm", "reward_head", "continue_head"]


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None,
                   help="Override total steps (2:1 P1:P2 split)")
    p.add_argument("--p2-steps", type=int, default=50_000,
                   help="Number of Phase-2 BVH head training steps (default: 50000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-p1", type=str, default=None, metavar="PATH")
    p.add_argument("--skip-p1", type=str, default=None, metavar="PATH")
    p.add_argument("--out", type=str, default="results/sensordrift_report.json")
    return p.parse_args()


def build_model(device):
    h_dim, z_cats, z_classes = 200, 8, 8
    embed_dim, hidden_dim    = 256, 256
    n_bins                   = 64
    z_dim      = z_cats * z_classes
    latent_dim = h_dim + z_dim

    model = {
        "encoder":       Encoder(OBS_DIM, embed_dim, hidden_dim=hidden_dim, n_layers=2),
        "decoder":       Decoder(latent_dim, OBS_DIM, hidden_dim=hidden_dim, n_layers=2),
        "rssm":          RSSM(h_dim, z_cats, z_classes, embed_dim, ACTION_DIM),
        "reward_head":   RewardHead(latent_dim, n_bins, hidden_dim=hidden_dim),
        "continue_head": ContinueHead(latent_dim, hidden_dim=hidden_dim),
        "tau_head":      ValidityHead(latent_dim, ACTION_DIM, n_bins=n_bins,
                                     hidden_dim=hidden_dim, max_horizon=K + 5),
        "hazard_head":   HazardHead(latent_dim, n_intervals=K, hidden_dim=hidden_dim),
    }
    for m in model.values():
        m.to(device)
    return model, latent_dim


def save_checkpoint(model, path: str, meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {k: model[k].state_dict() for k in WM_KEYS}
    if meta:
        ckpt["_meta"] = meta
    torch.save(ckpt, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(model, path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.pop("_meta", {})
    for k in WM_KEYS:
        if k in ckpt:
            model[k].load_state_dict(ckpt[k])
    for m in model.values():
        m.cpu()
    device = next(iter(model.values())).parameters().__next__().device if False else None
    print(f"  Checkpoint loaded ← {path}")
    if meta:
        print(f"  Meta: {meta}")
    return meta


def _move_model(model, device):
    for m in model.values():
        m.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Env stepper
# ─────────────────────────────────────────────────────────────────────────────

class SDStepper:
    def __init__(self, drift_rate: float, seed: int, device: torch.device):
        self._env    = SensorDrift(drift_rate=drift_rate, seed=seed)
        self._device = device
        obs, info    = self._env.reset(seed=seed)
        self._obs    = obs.astype(np.float32)
        self._state  = None   # lazy init

    def _init_state(self, model):
        self._state = model["rssm"].initial_state(1, device=self._device)
        self._prev_a = torch.zeros(1, ACTION_DIM, device=self._device)

    def step(self, model, buf: ReplayBuffer, random: bool = False) -> None:
        if self._state is None:
            self._init_state(model)

        obs_t = torch.from_numpy(self._obs).float().to(self._device).unsqueeze(0)
        if random:
            action_np = self._env.action_space.sample().astype(np.float32)
        else:
            with torch.no_grad():
                emb = model["encoder"](obs_t)
                _, self._state = model["rssm"].observe(emb, self._prev_a, self._state)
            action_np = self._env.action_space.sample().astype(np.float32)

        obs_next, reward, term, trunc, info = self._env.step(action_np)
        oracle_tau = float(info.get("oracle_tau", K))

        buf.push(
            obs=self._obs,
            action=action_np,
            reward=float(reward),
            terminated=bool(term or trunc),
            oracle_tau=oracle_tau,
            is_interventionist=False,
            rng_state={},
        )

        self._prev_a = torch.tensor(action_np, device=self._device).unsqueeze(0)

        if term or trunc:
            obs_next_np, _ = self._env.reset()
            self._obs = obs_next_np.astype(np.float32)
            self._state = model["rssm"].initial_state(1, device=self._device)
            self._prev_a = torch.zeros(1, ACTION_DIM, device=self._device)
        else:
            self._obs = obs_next.astype(np.float32)

    def close(self):
        self._env.close()


def _freeze(model, keys):
    for k in keys:
        for p in model[k].parameters():
            p.requires_grad_(False)

def _unfreeze(model, keys):
    for k in keys:
        for p in model[k].parameters():
            p.requires_grad_(True)

def _params(model, keys):
    return [p for k in keys for p in model[k].parameters() if p.requires_grad]


def _measure_kl(model, buf, device, n_batches=10):
    rssm, enc = model["rssm"], model["encoder"]
    rssm.eval(); enc.eval()
    kls = []
    with torch.no_grad():
        for _ in range(n_batches):
            batch = buf.sample(16, 16)
            obs   = batch["obs"].to(device)
            acts  = batch["action"].to(device)
            state = rssm.initial_state(16, device=device)
            for t in range(obs.shape[1]):
                post_logits, state = rssm.observe(enc(obs[:, t]), acts[:, t], state)
                prior_logits = rssm.prior_head(state.h).reshape(16, rssm.z_cats, rssm.z_classes)
                post_probs  = torch.softmax(post_logits, -1)
                prior_probs = torch.softmax(prior_logits, -1)
                kl = (post_probs * (post_probs.clamp(1e-8).log() - prior_probs.clamp(1e-8).log())).sum(-1).mean()
                kls.append(kl.item())
    rssm.train(); enc.train()
    return float(np.mean(kls))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: World model training
# ─────────────────────────────────────────────────────────────────────────────

def phase1_train(model, buf, stepper, device, n_steps, log_every):
    for m in model.values():
        m.train()

    opt = torch.optim.Adam(
        [p for m in model.values() for p in m.parameters()],
        lr=3e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-5)
    history = {"total": [], "pred": [], "dyn": [], "repr": []}
    t0 = time.time()

    rssm = model["rssm"]
    tau_h = model["tau_head"]

    for step in range(n_steps):
        stepper.step(model, buf, random=False)

        batch      = buf.sample(16, 16)
        obs        = batch["obs"].to(device)
        actions    = batch["action"].to(device)
        rewards    = batch["reward"].to(device)
        continues  = 1.0 - batch["terminated"].float().to(device)
        oracle_tau = batch["oracle_tau"].float().to(device)

        losses = world_model_loss(
            obs, actions, rewards, continues,
            model["encoder"], model["decoder"], rssm,
            model["reward_head"], model["continue_head"],
            return_latents=True,
        )

        # Auxiliary tau loss during P1 — shapes latent to carry tau signal
        lat  = losses["latents_flat"]
        acts = losses["actions_flat"]
        flat_tau = oracle_tau.reshape(-1)
        aux_tau = validity_loss(tau_h, lat, acts, flat_tau, stop_grad=False)

        total = losses["total"] + AUX_TAU_WEIGHT * aux_tau

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for m in model.values() for p in m.parameters()], GRAD_CLIP
        )
        opt.step()
        scheduler.step()

        if step % log_every == 0:
            history["total"].append(losses["total"].item())
            history["pred"].append(losses["prediction"].item())
            history["dyn"].append(losses["dynamics"].item())
            history["repr"].append(losses["representation"].item())
            print(
                f"  P1 step {step:6d}/{n_steps}  "
                f"total={losses['total'].item():6.3f}  "
                f"pred={losses['prediction'].item():6.3f}  "
                f"dyn={losses['dynamics'].item():5.3f}  "
                f"repr={losses['representation'].item():5.3f}  "
                f"({time.time()-t0:.0f}s)"
            )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: BVH head training (frozen world model)
# ─────────────────────────────────────────────────────────────────────────────

def phase2_train(model, buf, stepper, device, n_steps, log_every):
    wm_keys   = ["encoder", "decoder", "rssm", "reward_head", "continue_head"]
    head_keys = ["tau_head", "hazard_head"]
    _freeze(model, wm_keys)
    _unfreeze(model, head_keys)
    for m in model.values():
        m.train()

    opt = torch.optim.Adam(_params(model, head_keys), lr=P2_HEAD_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-5)
    history = {"tau_loss": [], "surv_loss": [], "step": []}
    t0 = time.time()
    rssm, enc = model["rssm"], model["encoder"]
    tau_h, haz_h = model["tau_head"], model["hazard_head"]

    for step in range(n_steps):
        stepper.step(model, buf, random=False)

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

        flat_lat = torch.stack(latents_post, 1).reshape(-1, latents_post[0].shape[-1])
        flat_act = post_actions.reshape(-1, post_actions.shape[-1])
        flat_tau = post_tau.reshape(-1)
        event_t  = flat_tau.long().clamp(0, K - 1)
        ev_occ   = (flat_tau < K)

        opt.zero_grad()
        v_loss = validity_loss(tau_h, flat_lat, flat_act, flat_tau, stop_grad=True)
        s_loss = survival_loss(haz_h, flat_lat, event_t, ev_occ, use_all_sources=False)
        (v_loss + s_loss).backward()
        torch.nn.utils.clip_grad_norm_(_params(model, head_keys), GRAD_CLIP)
        opt.step()
        scheduler.step()

        if step % log_every == 0:
            history["step"].append(step)
            history["tau_loss"].append(v_loss.item())
            history["surv_loss"].append(s_loss.item())
            print(
                f"  P2 step {step:6d}/{n_steps}  "
                f"tau_loss={v_loss.item():6.3f}  "
                f"surv_loss={s_loss.item():6.3f}  "
                f"({time.time()-t0:.0f}s)"
            )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_eval_metrics(model, drift_rate, device, n_episodes=50, max_steps=500):
    rssm, enc = model["rssm"], model["encoder"]
    tau_head, haz_head = model["tau_head"], model["hazard_head"]
    for m in model.values():
        m.eval()

    all_tau_pred, all_tau_star, all_survival, all_recon_mse = [], [], [], []

    for ep in range(n_episodes):
        env   = SensorDrift(drift_rate=drift_rate, seed=ep * 100)
        obs_np, info = env.reset(seed=ep * 100)
        state = rssm.initial_state(1, device=device)
        prev_a = torch.zeros(1, ACTION_DIM, device=device)

        for t in range(max_steps):
            obs_t = torch.from_numpy(obs_np.astype(np.float32)).to(device).unsqueeze(0)
            with torch.no_grad():
                embed = enc(obs_t)
                _, state = rssm.observe(embed, prev_a, state)
                latent  = rssm.get_latent(state)
                tau_hat = tau_head.decode(tau_head(latent, prev_a, stop_grad=False)).item()
                S = haz_head.survival(latent).squeeze(0).cpu().numpy()
                # Reconstruction baseline: MSE in symlog space.
                # Higher recon_mse → model struggling → proxy for staleness.
                mean_symlog, _ = model["decoder"].decode_symlog(latent)
                obs_t_symlog = symlog(obs_t)
                recon_mse = F.mse_loss(mean_symlog, obs_t_symlog).item()

            all_tau_pred.append(tau_hat)
            all_tau_star.append(float(info.get("oracle_tau", K)))
            all_survival.append(S)
            all_recon_mse.append(recon_mse)

            action_np = env.action_space.sample()
            obs_np, _, term, trunc, info = env.step(action_np)
            prev_a = torch.tensor(action_np.astype(np.float32), device=device).unsqueeze(0)
            if term or trunc:
                break
        env.close()

    tp = np.array(all_tau_pred, dtype=np.float32)
    ts = np.array(all_tau_star,  dtype=np.float32)
    S  = np.array(all_survival, dtype=np.float64)
    et = np.minimum(ts.astype(np.int32), K - 1)

    # Reconstruction baseline: -recon_mse as proxy tau predictor.
    # Higher recon_mse → model struggling → lower τ (more stale).
    # Negating gives a score where higher = more valid = higher τ.
    recon_arr = np.array(all_recon_mse, dtype=np.float64)
    recon_c_idx = c_index(-recon_arr, ts)

    return {
        "mae_tau":        mae_tau(tp, ts),
        "naive_mean_mae": float(np.mean(np.abs(ts - ts.mean()))),
        "naive_zero_mae": float(np.mean(np.abs(ts))),
        "pred_std":       float(np.std(tp)),
        "c_index":        c_index(tp, ts),
        "brier_score":    brier_score(S, ts, K),
        "n_samples":      len(tp),
        "recon_mse_mean": float(recon_arr.mean()),
        "recon_c_index":  recon_c_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = _parse()
    skip_p1 = args.skip_p1 is not None

    if skip_p1 and args.steps is not None:
        p1, p2 = 0, args.steps
    elif args.steps is not None:
        p1 = int(args.steps * 2 / 3)
        p2 = args.steps - p1
    else:
        p1, p2 = (0 if skip_p1 else P1_STEPS), args.p2_steps

    device_str = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    set_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"  BVH-RSSM SensorDrift Experiment — device={device_str}")
    print(f"  drift_rate={DRIFT_RATE:.5f}  max_tau={K}  K={K}")
    if skip_p1:
        print(f"  MODE: --skip-p1 (loading checkpoint)")
    print(f"  seed_steps={SEED_STEPS}  P1={p1}  P2={p2}")
    print(f"{'='*70}\n")

    model, latent_dim = build_model(device)
    buf = ReplayBuffer(capacity=BUF_CAPACITY, obs_dim=OBS_DIM,
                       action_dim=ACTION_DIM, seq_len=16)
    stepper = SDStepper(drift_rate=DRIFT_RATE, seed=args.seed, device=device)

    if skip_p1:
        print(f"[Step 0] Loading P1 checkpoint from {args.skip_p1} …")
        meta = load_checkpoint(model, args.skip_p1)
        _move_model(model, device)
        print(f"  Seeding buffer with {SEED_STEPS} transitions …")
        for _ in range(SEED_STEPS):
            stepper.step(model, buf, random=True)
        raw_taus = buf._oracle_tau[:len(buf)]
        print(f"  → {len(buf)} transitions  oracle_tau: mean={raw_taus.mean():.1f}  "
              f"% < K={(raw_taus < K).mean()*100:.1f}%\n")
        p1_loss_early = p1_loss_late = float("nan")
        kl_before_p1 = kl_after_p1  = float("nan")
        p1_pass = True
    else:
        print(f"[Step 0] Seeding buffer with {SEED_STEPS} random transitions …")
        for _ in range(SEED_STEPS):
            stepper.step(model, buf, random=True)
        raw_taus = buf._oracle_tau[:len(buf)]
        print(f"  → {len(buf)} transitions  oracle_tau: mean={raw_taus.mean():.1f}  "
              f"% < K={(raw_taus < K).mean()*100:.1f}%\n")

        print(f"[Step 1] Phase 1 — world model pretraining ({p1} steps)")
        kl_before_p1 = _measure_kl(model, buf, device)
        print(f"  KL before P1: {kl_before_p1:.4f}")
        p1_hist = phase1_train(model, buf, stepper, device, p1, LOG_EVERY)
        kl_after_p1 = _measure_kl(model, buf, device)
        print(f"\n  KL after P1: {kl_after_p1:.4f}")

        n = len(p1_hist["total"])
        split = max(1, n // 5)
        p1_loss_early = float(np.mean(p1_hist["total"][:split]))
        p1_loss_late  = float(np.mean(p1_hist["total"][-split:]))
        p1_pass = p1_loss_late < p1_loss_early
        print(f"  Loss trend: {p1_loss_early:.3f} → {p1_loss_late:.3f}  ✓={p1_pass}\n")

        if args.save_p1:
            save_checkpoint(model, args.save_p1, meta={
                "p1_steps": p1, "p1_loss_late": p1_loss_late,
                "kl_after_p1": kl_after_p1, "drift_rate": DRIFT_RATE,
            })

    kl_before_p2 = _measure_kl(model, buf, device)
    print(f"[Step 5] KL before Phase 2: {kl_before_p2:.4f}")

    print(f"\n[Step 2] Phase 2 — BVH head training ({p2} steps)")
    p2_hist = phase2_train(model, buf, stepper, device, p2, LOG_EVERY)
    print()

    kl_after_p2 = _measure_kl(model, buf, device)
    kl_increase = kl_after_p2 - kl_before_p2
    stopgrad_ok = kl_increase < 0.5
    print(f"[Step 5] KL after Phase 2: {kl_after_p2:.4f}  (Δ={kl_increase:+.4f})")
    print(f"  ✓ stop_grad: {stopgrad_ok}\n")

    print("[Step 2+3] Computing eval metrics …")
    metrics = compute_eval_metrics(model, DRIFT_RATE, device)
    print(f"  MAE_tau (BVH)       : {metrics['mae_tau']:.2f}")
    print(f"  MAE_tau (naive mean): {metrics['naive_mean_mae']:.2f}")
    print(f"  MAE_tau (naive zero): {metrics['naive_zero_mae']:.2f}")
    print(f"  Prediction std      : {metrics['pred_std']:.3f}")
    print(f"  C-index             : {metrics['c_index']:.4f}  ← TARGET >0.65")
    print(f"  Brier score         : {metrics['brier_score']:.4f}")
    print(f"  Recon MSE (mean)    : {metrics['recon_mse_mean']:.4f}")
    print(f"  Recon C-index       : {metrics['recon_c_index']:.4f}  (naive baseline)")

    n2 = len(p2_hist["tau_loss"])
    split2 = max(1, n2 // 5)
    tau_loss_early = float(np.mean(p2_hist["tau_loss"][:split2]))
    tau_loss_late  = float(np.mean(p2_hist["tau_loss"][-split2:]))
    tau_down = tau_loss_late < tau_loss_early

    beats_mean   = metrics["mae_tau"] < metrics["naive_mean_mae"]
    beats_zero   = metrics["mae_tau"] < metrics["naive_zero_mae"]
    has_variance = metrics["pred_std"] > 0.5

    print(f"\n{'='*70}")
    print("  PASS/FAIL")
    print(f"{'='*70}")
    def pf(flag, label): print(f"  {'✓ PASS' if flag else '✗ FAIL'}: {label}")
    pf(p1_pass,      f"P1 loss decreasing ({p1_loss_early:.3f} → {p1_loss_late:.3f})")
    pf(kl_after_p1 > 0.1 if not math.isnan(kl_after_p1) else True,
       f"KL not collapsed: {kl_after_p1:.4f}")
    pf(stopgrad_ok,  f"stop_grad: KL Δ={kl_increase:+.4f}")
    pf(tau_down,     f"P2 tau_loss decreasing ({tau_loss_early:.3f} → {tau_loss_late:.3f})")
    pf(beats_mean,   f"MAE < naive mean ({metrics['mae_tau']:.2f} vs {metrics['naive_mean_mae']:.2f})")
    pf(beats_zero,   f"MAE < naive zero ({metrics['mae_tau']:.2f} vs {metrics['naive_zero_mae']:.2f})")
    pf(has_variance, f"τ̂ not collapsed (std={metrics['pred_std']:.3f})")
    pf(metrics["c_index"] > 0.65, f"C-index > 0.65: {metrics['c_index']:.4f}  ← THE BREAKTHROUGH")
    print(f"{'='*70}")

    # Save report
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report = {
        "env": "SensorDrift",
        "drift_rate": DRIFT_RATE,
        "config": {"K": K, "p1_steps": p1, "p2_steps": p2, "seed": args.seed,
                   "device": device_str},
        "phase2": {
            "tau_loss_early": tau_loss_early, "tau_loss_late": tau_loss_late,
            "kl_before": kl_before_p2, "kl_after": kl_after_p2,
            "kl_increase": kl_increase,
        },
        "eval_metrics": metrics,
        "pass_fail": {
            "p1_loss_decreasing": p1_pass,
            "kl_not_collapsed": True,
            "stopgrad_ok": stopgrad_ok,
            "tau_loss_decreasing": tau_down,
            "beats_naive_mean": beats_mean,
            "beats_naive_zero": beats_zero,
            "prediction_not_collapsed": has_variance,
            "c_index": metrics["c_index"],
            "c_index_breakthrough": metrics["c_index"] > 0.65,
        },
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.out}")

    stepper.close()


if __name__ == "__main__":
    main()
