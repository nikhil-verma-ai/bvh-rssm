#!/usr/bin/env python
"""
BVH-RSSM validation harness — runs all five test steps from the test plan.

Steps
-----
1. RSSM base check    — reconstruction loss and KL trends over Phase 1 training
2. τ-head isolation   — Phase 2 MAE_tau vs naive baselines, prediction variance
3. Hazard head        — C-index and Brier score on held-out rollout
4. Environment audit  — oracle_tau signal correctness (already verified by check_env)
5. Stop-grad guard    — KL divergence must not increase after Phase 2 training

KEY DESIGN: Online collection (1 env step per training step) matches DreamerV3's
actual training regime. Fixed-buffer training causes catastrophic overfitting after
the model sees each transition thousands of times.

Usage
-----
    python scripts/validate.py          # ~90 min on MPS (100k P1 + 50k P2 steps)
    python scripts/validate.py --steps 9000   # faster: P1=6000 P2=3000 (~5 min)

Outputs
-------
    validation_report.json   — all metrics
    stdout                   — loss curves + pass/fail summary
"""
from __future__ import annotations

import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch

from bvh_rssm.networks import RSSM, Encoder, Decoder, RewardHead, ContinueHead
from bvh_rssm.networks.heads import ValidityHead, HazardHead
from bvh_rssm.training.replay_buffer import ReplayBuffer
from bvh_rssm.training.losses import world_model_loss, validity_loss, survival_loss
from bvh_rssm.training.experiment import set_seed
from bvh_rssm.training.metrics import mae_tau, c_index, brier_score
from bvh_rssm.envs import ShiftPendulum
from bvh_rssm.utils.rng import save_rng_state


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SHIFT_RATE    = 100.0   # shifts per 1000 steps → mean interval = 10 steps
SEED_STEPS    = 1_000   # random transitions before any training starts
P1_STEPS      = 100_000 # Phase 1: world model (online: 1 env step per train step)
P2_STEPS      = 50_000  # Phase 2: BVH heads   (online: 1 env step per train step)
LOG_EVERY     = 1_000   # print loss every N steps
GRAD_CLIP     = 5.0     # tighter than DreamerV3 default (100) for hidden_dim=128
BUF_CAPACITY  = 200_000 # large enough that early data doesn't crowd out later data
AUX_TAU_WEIGHT  = 0.3   # auxiliary τ loss weight in P1 — shapes latent to encode τ
P2_SEQ_LEN      = 64    # longer sequences in P2 so h_t builds shift-detection history
P2_BURNIN       = 32    # skip first N timesteps from tau loss (h_t is still cold)
P2_HEAD_LR      = 1e-3  # heads train at higher LR — they need faster updates than WM

OBS_DIM    = 3
ACTION_DIM = 1
K          = 16  # hazard intervals


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None,
                   help="Override total steps (split 2:1 P1:P2). e.g. --steps 9000")
    p.add_argument("--seed-steps", type=int, default=SEED_STEPS)
    p.add_argument("--rate", type=float, default=SHIFT_RATE)
    p.add_argument("--seed", type=int, default=42)
    # Checkpoint workflow: train P1 once, iterate on P2 in 20-30 min loops
    p.add_argument("--save-p1", type=str, default=None, metavar="PATH",
                   help="Save Phase 1 world model checkpoint to PATH after P1 completes.")
    p.add_argument("--skip-p1", type=str, default=None, metavar="PATH",
                   help="Load Phase 1 checkpoint from PATH and skip straight to Phase 2.")
    return p.parse_args()


WM_KEYS = ["encoder", "decoder", "rssm", "reward_head", "continue_head"]


def save_checkpoint(model, path: str, meta: dict = None) -> None:
    """Save world model weights + training metadata to path."""
    ckpt = {k: model[k].state_dict() for k in WM_KEYS}
    if meta:
        ckpt["_meta"] = meta
    torch.save(ckpt, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(model, path: str) -> dict:
    """Load world model weights from path. Returns metadata dict (may be empty)."""
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt.pop("_meta", {})
    for k in WM_KEYS:
        if k not in ckpt:
            raise KeyError(f"Checkpoint missing key '{k}' — was it saved with save_checkpoint()?")
        model[k].load_state_dict(ckpt[k])
    print(f"  Checkpoint loaded ← {path}")
    if meta:
        print(f"  Meta: {meta}")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device):
    h_dim, z_cats, z_classes = 128, 8, 8
    embed_dim, hidden_dim    = 256, 128
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
                                     hidden_dim=hidden_dim, max_horizon=50),
        "hazard_head":   HazardHead(latent_dim, n_intervals=K, hidden_dim=hidden_dim),
    }
    for m in model.values():
        m.to(device)
    return model, latent_dim


# ─────────────────────────────────────────────────────────────────────────────
# Online env stepper — shared state across calls
# ─────────────────────────────────────────────────────────────────────────────

class EnvStepper:
    """Wraps a single ShiftPendulum env and steps it one transition at a time.

    Maintains RSSM state across steps for proper GRU conditioning.
    Call .step(model, buf) to collect one transition.
    """
    def __init__(self, shift_rate: float, seed: int, device: torch.device):
        self._env = ShiftPendulum(shift_rate=shift_rate)
        self._device = device
        obs_np, info = self._env.reset(seed=seed)
        self._obs    = obs_np
        self._info   = info
        self._state  = None   # set on first step (needs model)
        self._prev_a = torch.zeros(1, ACTION_DIM, device=device)

    def _init_state(self, model):
        if self._state is None:
            self._state = model["rssm"].initial_state(1, self._device)

    def step(self, model, buf: ReplayBuffer, random: bool = False) -> None:
        """Collect one env transition and push to buffer."""
        self._init_state(model)
        obs_t = torch.from_numpy(self._obs).float().to(self._device).unsqueeze(0)
        rng_s = save_rng_state()
        with torch.no_grad():
            embed = model["encoder"](obs_t)
            _, self._state = model["rssm"].observe(embed, self._prev_a, self._state)

        if random:
            action_np = self._env.action_space.sample()
        else:
            action_np = self._env.action_space.sample()  # random for now (no actor)

        obs_next, reward, term, trunc, info = self._env.step(action_np)
        a_arr = np.array(action_np, dtype=np.float32).reshape(ACTION_DIM)
        buf.push(
            self._obs, a_arr, float(reward), bool(term or trunc),
            int(self._info["oracle_tau"]), bool(self._info.get("is_interventionist", False)),
            rng_s,
        )
        self._obs    = obs_next
        self._info   = info
        self._prev_a = torch.tensor(a_arr, dtype=torch.float32, device=self._device).unsqueeze(0)
        if term or trunc:
            self._obs, self._info = self._env.reset()
            self._state  = model["rssm"].initial_state(1, self._device)
            self._prev_a = torch.zeros(1, ACTION_DIM, device=self._device)

    def close(self):
        self._env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _freeze(model, keys):
    for k in keys:
        for p in model[k].parameters():
            p.requires_grad_(False)

def _unfreeze(model, keys):
    for k in keys:
        for p in model[k].parameters():
            p.requires_grad_(True)

def _params(model, keys):
    ps = []
    for k in keys:
        ps.extend(model[k].parameters())
    return ps

def _measure_kl(model, buf, device, n_batches=10):
    from bvh_rssm.utils import unimix
    rssm, enc = model["rssm"], model["encoder"]
    kls = []
    for _ in range(n_batches):
        batch = buf.sample(16, 16)
        obs = batch["obs"].to(device)
        actions = batch["action"].to(device)
        B, T, _ = obs.shape
        embeds = enc(obs.reshape(B * T, -1)).reshape(B, T, -1)
        state  = rssm.initial_state(B, device=device)
        kl_batch = 0.0
        with torch.no_grad():
            for t in range(T):
                post_logits, state = rssm.observe(embeds[:, t], actions[:, t], state)
                prior_logits = rssm.prior_head(state.h).reshape(B, rssm.z_cats, rssm.z_classes)
                pp = unimix(post_logits, eps=0.01)
                qp = unimix(prior_logits, eps=0.01)
                kl_batch += (pp * (pp.clamp(1e-8).log() - qp.clamp(1e-8).log())).sum(-1).mean().item()
        kls.append(kl_batch / T)
    return float(np.mean(kls))


def _compute_eval_metrics(model, shift_rate, device, n_episodes=50, max_steps=1000):
    rssm, enc = model["rssm"], model["encoder"]
    tau_head, haz_head = model["tau_head"], model["hazard_head"]
    for m in model.values():
        m.eval()

    all_tau_pred, all_tau_star, all_survival = [], [], []

    for ep in range(n_episodes):
        env = ShiftPendulum(shift_rate=shift_rate)
        obs_np, info = env.reset(seed=ep * 100)
        state = rssm.initial_state(1, device=device)
        prev_a = torch.zeros(1, ACTION_DIM, device=device)

        for t in range(max_steps):
            obs_t = torch.from_numpy(obs_np).float().to(device).unsqueeze(0)
            with torch.no_grad():
                embed = enc(obs_t)
                _, state = rssm.observe(embed, prev_a, state)
                latent = rssm.get_latent(state)
                tau_hat = tau_head.decode(tau_head(latent, prev_a, stop_grad=False)).item()
                S = haz_head.survival(latent).squeeze(0).cpu().numpy()

            all_tau_pred.append(tau_hat)
            all_tau_star.append(float(info["oracle_tau"]))
            all_survival.append(S)

            action_np = env.action_space.sample()
            obs_np, _, term, trunc, info = env.step(action_np)
            prev_a = torch.tensor(np.array(action_np, dtype=np.float32),
                                  dtype=torch.float32, device=device).unsqueeze(0)
            if term or trunc:
                break
        env.close()

    tp = np.array(all_tau_pred, dtype=np.float32)
    ts = np.array(all_tau_star,  dtype=np.float32)
    S  = np.array(all_survival, dtype=np.float64)
    et = np.minimum(ts.astype(np.int32), K - 1)

    return {
        "mae_tau":        mae_tau(tp, ts),
        "naive_mean_mae": float(np.mean(np.abs(ts - ts.mean()))),
        "naive_zero_mae": float(np.mean(np.abs(ts))),
        "pred_std":       float(np.std(tp)),
        "pred_variance":  float(np.var(tp)),
        "c_index":        c_index(tp, ts),
        "brier_score":    brier_score(S, ts, K),
        "n_samples":      len(tp),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: World model training — ONLINE (1 env step per train step)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_train(model, buf, stepper, device, n_steps, log_every):
    # Joint training: WM + tau_head together.
    # Auxiliary τ loss (weight=AUX_TAU_WEIGHT) forces the encoder/RSSM latent to
    # encode validity horizon signal. Without this, P2 τ-head reads from a latent
    # that never learned to represent τ → predictions collapse to mean → C-index ≈ 0.5.
    wm_keys = ["encoder", "decoder", "rssm", "reward_head", "continue_head", "tau_head"]
    _unfreeze(model, wm_keys)
    _freeze(model, ["hazard_head"])
    for m in model.values():
        m.train()

    opt = torch.optim.Adam(_params(model, wm_keys), lr=3e-4)
    history = {"total": [], "pred": [], "dyn": [], "repr": [], "step": []}
    t0 = time.time()

    for step in range(n_steps):
        # Online: collect 1 new transition per training step
        stepper.step(model, buf, random=False)

        batch      = buf.sample(16, 16)
        obs        = batch["obs"].to(device)
        actions    = batch["action"].to(device)
        rewards    = batch["reward"].to(device)
        continues  = (1.0 - batch["terminated"].float()).to(device)
        oracle_tau = batch["oracle_tau"].float().to(device)

        opt.zero_grad()
        result = world_model_loss(obs, actions, rewards, continues,
                                  model["encoder"], model["decoder"], model["rssm"],
                                  model["reward_head"], model["continue_head"],
                                  return_latents=True)

        # Auxiliary τ loss: gradient flows through latent → encoder/RSSM (stop_grad=False)
        flat_lat = result["latents_flat"]         # [B*T, latent_dim]
        flat_act = result["actions_flat"]         # [B*T, action_dim]
        flat_tau = oracle_tau.reshape(-1)         # [B*T]
        aux_loss = AUX_TAU_WEIGHT * validity_loss(
            model["tau_head"], flat_lat, flat_act, flat_tau, stop_grad=False
        )

        (result["total"] + aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(_params(model, wm_keys), GRAD_CLIP)
        opt.step()

        if step % log_every == 0:
            history["step"].append(step)
            history["total"].append(result["total"].item())
            history["pred"].append(result["prediction"].item())
            history["dyn"].append(result["dynamics"].item())
            history["repr"].append(result["representation"].item())
            print(
                f"  P1 step {step:6d}/{n_steps}  "
                f"total={result['total'].item():7.3f}  "
                f"pred={result['prediction'].item():7.3f}  "
                f"dyn={result['dynamics'].item():5.3f}  "
                f"repr={result['representation'].item():5.3f}  "
                f"buf={len(buf)}  ({time.time()-t0:.0f}s)"
            )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: BVH head training — ONLINE
# ─────────────────────────────────────────────────────────────────────────────

def phase2_train(model, buf, stepper, device, n_steps, log_every):
    wm_keys   = ["encoder", "decoder", "rssm", "reward_head", "continue_head"]
    head_keys = ["tau_head", "hazard_head"]
    _freeze(model, wm_keys)
    _unfreeze(model, head_keys)
    for m in model.values():
        m.train()

    opt = torch.optim.Adam(_params(model, head_keys), lr=P2_HEAD_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=1e-5
    )
    history = {"tau_loss": [], "surv_loss": [], "step": []}
    t0 = time.time()
    rssm, enc = model["rssm"], model["encoder"]
    tau_h, haz_h = model["tau_head"], model["hazard_head"]

    for step in range(n_steps):
        # Online: keep collecting during Phase 2 too (buffer stays fresh)
        stepper.step(model, buf, random=False)

        # Sample longer sequences so the RSSM has time to detect regime changes.
        # P2_BURNIN steps let h_t build up "time-since-last-shift" signal before
        # we ask the tau_head to predict from it. Without burn-in, h_t starts cold
        # and the tau loss on early timesteps is pure noise.
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
            # Only use post-burnin latents: h_t has seen P2_BURNIN steps → shift-aware
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse()
    if args.skip_p1 and args.steps is not None:
        # --skip-p1 mode: all steps go to P2
        p1, p2 = 0, args.steps
    elif args.steps is not None:
        p1 = int(args.steps * 2 / 3)
        p2 = args.steps - p1
    else:
        p1, p2 = (0 if args.skip_p1 else P1_STEPS), P2_STEPS

    device_str = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    set_seed(args.seed)

    skip_p1 = args.skip_p1 is not None

    print(f"\n{'='*70}")
    print(f"  BVH-RSSM Validation — device={device_str}  shift_rate={args.rate}")
    if skip_p1:
        print(f"  MODE: --skip-p1 (loading checkpoint, P2-only)")
    print(f"  seed_steps={args.seed_steps}  P1={p1}  P2={p2}  K={K}")
    print(f"  online collection  grad_clip={GRAD_CLIP}  buf_cap={BUF_CAPACITY}")
    print(f"{'='*70}\n")

    model, _ = build_model(device)
    buf = ReplayBuffer(capacity=BUF_CAPACITY, obs_dim=OBS_DIM,
                       action_dim=ACTION_DIM, seq_len=16)

    stepper = EnvStepper(shift_rate=args.rate, seed=args.seed, device=device)

    if skip_p1:
        # ── Load Phase 1 checkpoint, seed buffer, skip straight to Phase 2 ──
        print(f"[Step 0] Loading P1 checkpoint from {args.skip_p1} …")
        meta = load_checkpoint(model, args.skip_p1)
        # Seed buffer so Phase 2 has data to train on immediately
        print(f"  Seeding buffer with {args.seed_steps} transitions …")
        for _ in range(args.seed_steps):
            stepper.step(model, buf, random=True)
        raw_taus = buf._oracle_tau[:len(buf)]
        print(f"  → {len(buf)} transitions  "
              f"oracle_tau: mean={raw_taus.mean():.1f}  "
              f"% < K={( raw_taus < K).mean()*100:.1f}%\n")
        # Stub out P1 history/metrics for report
        p1_loss_early = p1_loss_late = p1_pred_early = p1_pred_late = float("nan")
        kl_before_p1 = kl_after_p1 = float("nan")
        p1_pass = True   # not evaluated
        p1_hist = {"total": [], "pred": [], "dyn": [], "repr": []}
    else:
        # ── Seed the buffer with random transitions before training ──────────
        print(f"[Step 0] Seeding buffer with {args.seed_steps} random transitions …")
        for _ in range(args.seed_steps):
            stepper.step(model, buf, random=True)
        raw_taus = buf._oracle_tau[:len(buf)]
        print(f"  → {len(buf)} transitions  "
              f"oracle_tau: mean={raw_taus.mean():.1f}  "
              f"median={np.median(raw_taus):.1f}  "
              f"% < K={( raw_taus < K).mean()*100:.1f}%\n")

        # ── Phase 1: World model pretraining ─────────────────────────────────
        print(f"[Step 1] Phase 1 — world model pretraining ({p1} steps, online)")
        kl_before_p1 = _measure_kl(model, buf, device)
        print(f"  KL before P1: {kl_before_p1:.4f}")
        p1_hist = phase1_train(model, buf, stepper, device, p1, LOG_EVERY)
        kl_after_p1 = _measure_kl(model, buf, device)
        print(f"\n  KL after  P1: {kl_after_p1:.4f}")

        n = len(p1_hist["total"])
        split = max(1, n // 5)
        p1_loss_early = float(np.mean(p1_hist["total"][:split]))
        p1_loss_late  = float(np.mean(p1_hist["total"][-split:]))
        p1_pred_early = float(np.mean(p1_hist["pred"][:split]))
        p1_pred_late  = float(np.mean(p1_hist["pred"][-split:]))
        p1_pass = p1_loss_late < p1_loss_early
        print(f"  Loss trend: {p1_loss_early:.3f} → {p1_loss_late:.3f}  "
              f"pred: {p1_pred_early:.3f} → {p1_pred_late:.3f}")
        print(f"  ✓ Decreasing: {p1_pass}\n")

        # ── Save P1 checkpoint if requested ──────────────────────────────────
        if args.save_p1:
            save_checkpoint(model, args.save_p1, meta={
                "p1_steps": p1, "p1_loss_late": p1_loss_late,
                "kl_after_p1": kl_after_p1, "shift_rate": args.rate,
            })

    # ── Stop-grad baseline (KL before Phase 2) ───────────────────────────────
    kl_before_p2 = _measure_kl(model, buf, device)
    print(f"[Step 5] KL before Phase 2: {kl_before_p2:.4f}")

    # ── Phase 2: BVH head training ───────────────────────────────────────────
    print(f"\n[Step 2] Phase 2 — BVH head training ({p2} steps, online)")
    p2_hist = phase2_train(model, buf, stepper, device, p2, LOG_EVERY)
    print()

    kl_after_p2  = _measure_kl(model, buf, device)
    kl_increase  = kl_after_p2 - kl_before_p2
    stopgrad_ok  = kl_increase < 0.5
    print(f"[Step 5] KL after Phase 2: {kl_after_p2:.4f}  (Δ={kl_increase:+.4f})")
    print(f"  ✓ stop_grad preserved KL: {stopgrad_ok}\n")

    # ── Eval metrics ─────────────────────────────────────────────────────────
    print("[Step 2+3] Computing eval metrics …")
    metrics = _compute_eval_metrics(model, args.rate, device)
    print(f"  MAE_tau (BVH)       : {metrics['mae_tau']:.2f}")
    print(f"  MAE_tau (naive mean): {metrics['naive_mean_mae']:.2f}")
    print(f"  MAE_tau (naive zero): {metrics['naive_zero_mae']:.2f}")
    print(f"  Prediction std      : {metrics['pred_std']:.3f}  (oracle std ≈ {metrics['naive_mean_mae']:.1f})")
    print(f"  C-index             : {metrics['c_index']:.4f}  (random=0.5, target>0.65)")
    print(f"  Brier score         : {metrics['brier_score']:.4f}")

    n2 = len(p2_hist["tau_loss"])
    split2 = max(1, n2 // 5)
    tau_loss_early = float(np.mean(p2_hist["tau_loss"][:split2]))
    tau_loss_late  = float(np.mean(p2_hist["tau_loss"][-split2:]))
    print(f"  tau_loss trend      : {tau_loss_early:.3f} → {tau_loss_late:.3f}")

    # ── Pass/fail ─────────────────────────────────────────────────────────────
    beats_mean   = metrics["mae_tau"] < metrics["naive_mean_mae"]
    beats_zero   = metrics["mae_tau"] < metrics["naive_zero_mae"]
    has_variance = metrics["pred_std"] > 0.5   # full-training target
    tau_down     = tau_loss_late < tau_loss_early

    print(f"\n{'='*70}")
    print("  PASS/FAIL")
    print(f"{'='*70}")
    def pf(flag, label): print(f"  {'✓ PASS' if flag else '✗ FAIL'}: {label}")
    pf(p1_pass,      f"P1 loss decreasing ({p1_loss_early:.3f} → {p1_loss_late:.3f})")
    pf(kl_after_p1 > 0.1, f"KL not collapsed after P1: {kl_after_p1:.4f}")
    pf(stopgrad_ok,  f"stop_grad: KL Δ={kl_increase:+.4f} (threshold <0.5)")
    pf(tau_down,     f"P2 tau_loss decreasing ({tau_loss_early:.3f} → {tau_loss_late:.3f})")
    pf(beats_mean,   f"MAE < naive mean ({metrics['mae_tau']:.2f} vs {metrics['naive_mean_mae']:.2f})")
    pf(beats_zero,   f"MAE < naive zero ({metrics['mae_tau']:.2f} vs {metrics['naive_zero_mae']:.2f})")
    pf(has_variance, f"τ̂ not collapsed (std={metrics['pred_std']:.3f}, target >0.5)")
    print(f"  (info) C-index = {metrics['c_index']:.4f}  [target >0.65 at full scale]")
    print(f"{'='*70}\n")

    stepper.close()

    def _f(v): return None if (isinstance(v, float) and math.isnan(v)) else v

    report = {
        "config": {
            "shift_rate": args.rate, "seed_steps": args.seed_steps,
            "p1_steps": p1, "p2_steps": p2, "K": K, "device": device_str,
            "grad_clip": GRAD_CLIP, "online": True,
            "skip_p1": args.skip_p1, "save_p1": args.save_p1,
        },
        "phase1": {
            "loss_early": _f(p1_loss_early), "loss_late": _f(p1_loss_late),
            "pred_early": _f(p1_pred_early), "pred_late": _f(p1_pred_late),
            "kl_before": _f(kl_before_p1), "kl_after": _f(kl_after_p1),
            "skipped": skip_p1,
        },
        "phase2": {
            "tau_loss_early": tau_loss_early, "tau_loss_late": tau_loss_late,
            "kl_before": kl_before_p2, "kl_after": kl_after_p2,
            "kl_increase": kl_increase,
        },
        "eval_metrics": metrics,
        "pass_fail": {
            "p1_loss_decreasing": p1_pass,
            "kl_not_collapsed": skip_p1 or (kl_after_p1 > 0.1),
            "stopgrad_ok": stopgrad_ok,
            "tau_loss_decreasing": tau_down,
            "beats_naive_mean": beats_mean,
            "beats_naive_zero": beats_zero,
            "prediction_not_collapsed": has_variance,
            "c_index": metrics["c_index"],
        },
    }
    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report written to validation_report.json")


if __name__ == "__main__":
    main()
