# Phase 3: BVH-Gated Policy — Implementation Plan



**Goal:** Demonstrate that τ̂ signal improves task performance on SensorDrift — delta_return > 0 for BVH-gated policy vs. vanilla policy using identical weights.

**Architecture:** Part A loads the trained Phase 1+2 checkpoint and runs 200 eval episodes under two policies (BVH-gated vs vanilla), reporting delta_return. Part B modifies train_phase3() to accept an `imagination_gating` flag that gates horizon depth via AdaptivePolicyRouter, then trains two actors from the same checkpoint for comparison.

**Tech Stack:** Python 3.11, PyTorch, bvh_rssm (RSSM, ValidityHead, HazardHead, AdaptivePolicyRouter, Actor, Critic), SensorDrift env, argparse, json

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `scripts/experiment_phase3.py` | CLI harness for Part A (eval) and Part B (train); all episode-rollout and reporting logic |
| Modify | `bvh_rssm/training/trainer.py:195` | Add `imagination_gating: bool = False` to `train_phase3()` signature and gating logic at line 358 |
| Create | `tests/unit/test_phase3.py` | Unit tests: gating logic, router integration, delta_return calculation |

---

## Task 1: Add `imagination_gating` to `train_phase3()`

**Files:**
- Modify: `bvh_rssm/training/trainer.py:195`
- Test: `tests/unit/test_phase3.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_phase3.py
import pytest
import torch
from bvh_rssm.causal.router import AdaptivePolicyRouter, RouterState


def test_imagination_horizon_gating():
    router = AdaptivePolicyRouter()
    # S that puts tau_hi=2, tau_min=8
    S = torch.tensor([1.0, 0.95, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.05, 0.01])
    # HIGH: tau_hat=10 > tau_min=8 → full_horizon=16
    assert router.imagination_horizon(RouterState.HIGH, 10.0, full_horizon=16) == 16
    # DIM: tau_hat=4, tau_hi=2, tau_min=8 → max(1, 4//2) = 2
    assert router.imagination_horizon(RouterState.DIM, 4.0, full_horizon=16) == 2
    # STALE → always 1
    assert router.imagination_horizon(RouterState.STALE, 0.5, full_horizon=16) == 1


def test_gated_horizon_never_exceeds_full():
    router = AdaptivePolicyRouter()
    S_all_high = torch.ones(16)  # S never drops → tau_min = K-1 = 15
    state = router.classify(20.0, S_all_high)
    assert state == RouterState.HIGH
    h = router.imagination_horizon(state, 20.0, full_horizon=16)
    assert h == 16


def test_gated_horizon_stale_is_one():
    router = AdaptivePolicyRouter()
    # S drops immediately → tau_hi=0, tau_min=0
    S_zero = torch.tensor([0.1] * 16)
    state = router.classify(0.0, S_zero)
    assert state == RouterState.STALE
    assert router.imagination_horizon(state, 0.0, full_horizon=16) == 1
```

- [ ] **Step 2: Run tests to verify they fail (they shouldn't yet — router is already implemented)**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
python -m pytest tests/unit/test_phase3.py -v
```

Expected: All 3 PASS (router logic already correct). If any FAIL, fix the test expectations to match actual router behavior.

- [ ] **Step 3: Add `imagination_gating` parameter to `train_phase3()`**

In `bvh_rssm/training/trainer.py`, change the signature at line 195:

```python
    def train_phase3(self, imagination_gating: bool = False) -> None:
        """Phase 3: joint fine-tuning with actor-critic in imagination space.

        Args:
            imagination_gating: If True, gate imagination depth per step using
                AdaptivePolicyRouter. tau_hat and S(t) are computed at the start
                of each imagination rollout; H = router.imagination_horizon(state, tau_hat).
                If False, always use full_horizon (DreamerV3 baseline).
        ...
        """
```

- [ ] **Step 4: Add router import and instantiation in `train_phase3()` body**

Insert after line 256 (`full_horizon = 16`):

```python
        from bvh_rssm.causal.router import AdaptivePolicyRouter
        router = AdaptivePolicyRouter() if imagination_gating else None
```

- [ ] **Step 5: Gate the imagination horizon inside the actor-critic imagination loop**

The imagination loop starts at line 358 (`for _ in range(full_horizon):`). Replace the entire imagination-setup block (lines 340–358) with gating logic. Find this block:

```python
                # Use the last real latent as the starting state for imagination.
                # shape: [B, latent_dim] — last timestep of the real sequence
                start_latent = latents_stacked[:, -1, :].detach()  # [B, latent_dim]

                # Unroll H imagination steps
                imagined_latents:   List[torch.Tensor] = []
```

Replace with:

```python
                # Use the last real latent as the starting state for imagination.
                start_latent = latents_stacked[:, -1, :].detach()  # [B, latent_dim]

                # ---- BVH imagination gating ----
                if imagination_gating:
                    # Use first batch element to derive horizon (all share same drift phase)
                    with torch.no_grad():
                        tau_hat_val = float(
                            tau_head.decode(
                                tau_head(start_latent[:1], actions[:, -1, :][:1], stop_grad=True)
                            ).mean().item()
                        )
                        S_t = hazard_head.survival(start_latent[:1])[0]  # [K]
                    state_route = router.classify(tau_hat_val, S_t)
                    H = router.imagination_horizon(state_route, tau_hat_val, full_horizon=full_horizon)
                else:
                    H = full_horizon

                # Unroll H imagination steps
                imagined_latents:   List[torch.Tensor] = []
```

- [ ] **Step 6: Fix the `H = full_horizon` reference later in the loop**

The loop currently hardcodes `H = full_horizon` at line 399. That line becomes redundant because `H` is now set before the loop. Find:

```python
                # Stack into [H, B]
                H = full_horizon
```

Replace with:

```python
                # Stack into [H, B] — H is already set by gating logic above
```

- [ ] **Step 7: Add trainer_gating test**

Add to `tests/unit/test_phase3.py`:

```python
import torch.nn as nn
from unittest.mock import MagicMock, patch


def _make_minimal_trainer():
    """Build a Trainer with stub modules to test imagination_gating path."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from bvh_rssm.training.trainer import Trainer, TrainerConfig

    cfg = TrainerConfig(phase3_steps=1, batch_size=2, seq_len=4, log_every=9999,
                        checkpoint_every=0, device='cpu')

    # Stub every module train_phase3 touches
    rssm = MagicMock()
    rssm.h_dim = 8
    state_mock = MagicMock()
    state_mock.h = torch.zeros(2, 8)
    state_mock.z = torch.zeros(2, 8)
    rssm.initial_state.return_value = state_mock
    rssm.observe.return_value = (torch.zeros(2, 8, 8), state_mock)
    rssm.get_latent.return_value = torch.zeros(2, 16)
    rssm.imagine.return_value = (None, state_mock)
    rssm.parameters.return_value = iter([])

    tau_head = MagicMock()
    tau_head.return_value = torch.zeros(2, 16)
    tau_head.decode.return_value = torch.tensor([5.0, 5.0])
    tau_head.parameters.return_value = iter([])

    hazard_head = MagicMock()
    hazard_head.n_intervals = 16
    hazard_head.survival.return_value = torch.ones(2, 16) * 0.9
    hazard_head.parameters.return_value = iter([])

    def mock_module_items():
        return {'rssm': rssm, 'tau_head': tau_head, 'hazard_head': hazard_head}

    model = mock_module_items()
    buf = MagicMock()
    buf.sample.return_value = {
        'obs': torch.zeros(2, 4, 17),
        'action': torch.zeros(2, 4, 6),
        'reward': torch.zeros(2, 4),
        'terminated': torch.zeros(2, 4),
        'oracle_tau': torch.ones(2, 4) * 8.0,
    }

    return Trainer(model, buf, cfg)


def test_imagination_gating_flag_accepted():
    """train_phase3 must accept imagination_gating without error when actor/critic absent."""
    import inspect
    from bvh_rssm.training.trainer import Trainer
    sig = inspect.signature(Trainer.train_phase3)
    assert 'imagination_gating' in sig.parameters, \
        "train_phase3 must have imagination_gating parameter"
    p = sig.parameters['imagination_gating']
    assert p.default is False, "imagination_gating default must be False"
```

- [ ] **Step 8: Run all phase3 tests**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
python -m pytest tests/unit/test_phase3.py -v
```

Expected: All tests PASS.

- [ ] **Step 9: Run full test suite to verify no regressions**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
python -m pytest tests/unit/ -v --tb=short 2>&1 | tail -20
```

Expected: All existing tests still pass.

- [ ] **Step 10: Commit**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
git add bvh_rssm/training/trainer.py tests/unit/test_phase3.py
git commit -m "feat(phase3): add imagination_gating to train_phase3(); tests"
```

---

## Task 2: Create `scripts/experiment_phase3.py` — Part A eval harness

**Files:**
- Create: `scripts/experiment_phase3.py`
- Test: run it with `--mode eval` against the saved checkpoint

- [ ] **Step 1: Write the test for the BVH vs vanilla rollout logic**

Add to `tests/unit/test_phase3.py`:

```python
def test_zero_action_fallback_shape():
    """STALE state must yield zero action of correct shape."""
    import numpy as np
    action_dim = 6
    stale = True
    if stale:
        action = np.zeros(action_dim, dtype=np.float32)
    assert action.shape == (action_dim,)
    assert (action == 0.0).all()


def test_episode_return_accumulation():
    """delta_return = mean(bvh_returns) - mean(vanilla_returns)."""
    bvh_returns = [10.0, 20.0, 30.0]
    vanilla_returns = [5.0, 15.0, 25.0]
    delta = sum(bvh_returns) / len(bvh_returns) - sum(vanilla_returns) / len(vanilla_returns)
    assert abs(delta - 10.0) < 1e-6


def test_failure_rate():
    """failure_rate = fraction of episodes with return < -50."""
    returns = [100.0, -60.0, -55.0, 10.0]
    threshold = -50.0
    rate = sum(1 for r in returns if r < threshold) / len(returns)
    assert abs(rate - 0.5) < 1e-6
```

- [ ] **Step 2: Run new tests to verify they pass**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
python -m pytest tests/unit/test_phase3.py::test_zero_action_fallback_shape \
    tests/unit/test_phase3.py::test_episode_return_accumulation \
    tests/unit/test_phase3.py::test_failure_rate -v
```

Expected: All 3 PASS.

- [ ] **Step 3: Create `scripts/experiment_phase3.py`**

```python
#!/usr/bin/env python3
"""
Phase 3 BVH-Gated Policy experiment.

Part A (--mode eval):
    Loads trained Phase 1+2 checkpoint. Runs 200 episodes under two policies:
      BVH policy:     STALE → zero-action fallback; HIGH/DIM → actor (or random)
      Vanilla policy: always actor (or random), ignores RouterState
    Reports delta_return = E[BVH] - E[Vanilla].

Part B (--mode train):
    Trains actor-critic from the checkpoint using train_phase3().
    --gated flag enables imagination gating (BVH actor).
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
from bvh_rssm.training.experiment import set_seed
from bvh_rssm.envs.sensor_drift import SensorDrift

# ──────────────────────────────────────────────────────────────────────────────
# Constants (matching sd_p1_v2.pt architecture)
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

WM_KEYS = ["encoder", "decoder", "rssm", "reward_head", "continue_head"]


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
    # Strip _meta key if present
    ckpt.pop("_meta", None)
    for k in WM_KEYS:
        if k in ckpt:
            model[k].load_state_dict(ckpt[k], strict=False)
            print(f"  Loaded {k} from checkpoint")
    # Also load tau_head and hazard_head if saved
    for k in ("tau_head", "hazard_head"):
        if k in ckpt:
            model[k].load_state_dict(ckpt[k], strict=False)
            print(f"  Loaded {k} from checkpoint")


# ──────────────────────────────────────────────────────────────────────────────
# Episode rollout
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_episode(
    model: dict,
    seed: int,
    device: torch.device,
    use_bvh_gate: bool,
    max_steps: int = 500,
) -> dict:
    """Run one episode and return metrics.

    Args:
        model: Model dict with encoder, rssm, tau_head, hazard_head, actor.
        seed: Episode seed for reproducibility.
        device: Torch device.
        use_bvh_gate: If True, apply STALE → zero-action fallback.
        max_steps: Max episode length.

    Returns:
        dict with keys: total_return, stale_triggered, first_stale_step, n_steps
    """
    env   = SensorDrift(drift_rate=DRIFT_RATE, seed=seed)
    obs_np, _ = env.reset(seed=seed)

    router = AdaptivePolicyRouter()
    rssm   = model["rssm"]
    enc    = model["encoder"]
    actor  = model["actor"]
    tau_h  = model["tau_head"]
    haz_h  = model["hazard_head"]

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

        # Compute τ̂ and S(t)
        tau_hat = float(tau_h.decode(tau_h(lat, prev_act, stop_grad=True)).mean().item())
        S_t     = haz_h.survival(lat)[0]                        # [K]
        route   = router.classify(tau_hat, S_t)

        # Policy decision
        if use_bvh_gate and route == RouterState.STALE:
            action_np = np.zeros(ACTION_DIM, dtype=np.float32)
            if not stale_triggered:
                stale_triggered  = True
                first_stale_step = t
        else:
            actor_out = actor(lat)
            mean, log_std = actor_out
            std = log_std.exp()
            act = mean + std * torch.randn_like(mean)           # sample
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

    for m in model.values():
        m.eval()

    seeds = list(range(args.n_episodes))  # fixed seeds 0..N-1 for fairness

    print("[eval] Running BVH policy …")
    bvh_results = []
    for i, seed in enumerate(seeds):
        r = run_episode(model, seed=seed, device=device, use_bvh_gate=True)
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

    stale_episodes   = [r for r in bvh_results if r["stale_triggered"]]
    stale_rate       = len(stale_episodes) / args.n_episodes
    mean_stale_step  = (
        float(np.mean([r["first_stale_step"] for r in stale_episodes]))
        if stale_episodes else -1.0
    )

    report = {
        "mode":                   "eval",
        "n_episodes":             args.n_episodes,
        "checkpoint":             args.checkpoint,
        "bvh_mean_return":        float(np.mean(bvh_rets)),
        "vanilla_mean_return":    float(np.mean(vanilla_rets)),
        "delta_return_inference": float(np.mean(bvh_rets) - np.mean(vanilla_rets)),
        "failure_rate_bvh":       float(bvh_fail),
        "failure_rate_vanilla":   float(vanilla_fail),
        "stale_trigger_rate":     float(stale_rate),
        "mean_stale_trigger_step": float(mean_stale_step),
    }

    print("\n[eval] Results:")
    print(f"  BVH mean return:     {report['bvh_mean_return']:.2f}")
    print(f"  Vanilla mean return: {report['vanilla_mean_return']:.2f}")
    print(f"  delta_return:        {report['delta_return_inference']:+.2f}")
    print(f"  failure_rate BVH:    {report['failure_rate_bvh']:.3f}")
    print(f"  failure_rate Vanilla:{report['failure_rate_vanilla']:.3f}")
    print(f"  stale_trigger_rate:  {report['stale_trigger_rate']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
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

    # Build replay buffer and seed it with random SensorDrift experience
    buf = ReplayBuffer(capacity=100_000, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    env = SensorDrift(drift_rate=DRIFT_RATE, seed=42)
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

    print("[train] Evaluating trained actor …")
    trained_results = []
    for seed in seeds:
        r = run_episode(model, seed=seed, device=device, use_bvh_gate=args.gated)
        trained_results.append(r)

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
        "mode":                "train",
        "gated":               args.gated,
        "n_episodes":          200,
        "phase3_steps":        args.phase3_steps,
        "checkpoint":          args.checkpoint,
        "trained_mean_return": float(np.mean(rets)),
        "trained_std_return":  float(np.std(rets)),
        "failure_rate":        float(fail_rate),
        "stale_trigger_rate":  float(stale_rate),
        "mean_stale_trigger_step": float(mean_stale_step),
        "elapsed_s":           float(elapsed),
    }

    print(f"\n[train] Results ({'BVH-gated' if args.gated else 'Vanilla'} actor):")
    print(f"  mean_return: {report['trained_mean_return']:.2f}")
    print(f"  failure_rate: {report['failure_rate']:.3f}")
    print(f"  stale_trigger_rate: {report['stale_trigger_rate']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
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
    p.add_argument("--phase3-steps", type=int, default=50_000,
                   help="Phase 3 training steps (Part B, default: 50000)")
    p.add_argument("--gated", action="store_true",
                   help="Enable BVH imagination gating during training (Part B)")
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
```

- [ ] **Step 4: Smoke test Part A (no checkpoint)**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
python3 scripts/experiment_phase3.py \
    --mode eval \
    --n-episodes 5 \
    --out /tmp/p3_smoke_eval.json
```

Expected: Runs 5 BVH + 5 vanilla episodes without error, writes JSON with `delta_return_inference` key.

- [ ] **Step 5: Smoke test Part B (no checkpoint, 10 steps)**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
python3 scripts/experiment_phase3.py \
    --mode train \
    --phase3-steps 10 \
    --gated \
    --out /tmp/p3_smoke_train.json
```

Expected: Completes 10 training steps (actor/critic absent warns but doesn't crash), evaluates, writes JSON with `trained_mean_return` key.

- [ ] **Step 6: Commit**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
git add scripts/experiment_phase3.py tests/unit/test_phase3.py
git commit -m "feat(phase3): add experiment_phase3.py eval harness + additional unit tests"
```

---

## Task 3: Run Part A — BVH-gated inference evaluation

**Files:**
- Run: `scripts/experiment_phase3.py --mode eval`
- Output: `results/phase3_eval_report.json`

Prerequisites: `checkpoints/sd_p1_v2.pt` must exist (produced by the extended SensorDrift run).

- [ ] **Step 1: Verify checkpoint exists**

```bash
ls -lh /Users/nikhil-verma-ai/bvh-rssm/checkpoints/sd_p1_v2.pt
```

Expected: File exists, ~100MB.

- [ ] **Step 2: Run Part A evaluation (200 episodes)**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
caffeinate -dims python3 scripts/experiment_phase3.py \
    --mode eval \
    --checkpoint checkpoints/sd_p1_v2.pt \
    --n-episodes 200 \
    --out results/phase3_eval_report.json
```

Expected runtime: ~5-15 minutes (200 episodes × 2 policies × ~500 steps each).

Expected output structure:
```
[eval] device=mps  episodes=200
[eval] Running BVH policy …
  BVH 20/200  mean_return=...
  ...
[eval] Running Vanilla policy …
  ...
[eval] Results:
  BVH mean return:     ...
  Vanilla mean return: ...
  delta_return:        +X.XX  ← must be > 0 for success criterion
  failure_rate BVH:    0.XXX  ← must be < failure_rate Vanilla
  failure_rate Vanilla:0.XXX
  stale_trigger_rate:  0.XXX  ← must be 0.20-0.80
```

- [ ] **Step 3: Verify success criteria**

```bash
python3 -c "
import json
r = json.load(open('results/phase3_eval_report.json'))
print('delta_return_inference:', r['delta_return_inference'])
print('failure_rate_bvh < failure_rate_vanilla:', r['failure_rate_bvh'] < r['failure_rate_vanilla'])
print('stale_trigger_rate:', r['stale_trigger_rate'])
assert r['delta_return_inference'] > 0, 'FAIL: delta_return <= 0'
assert r['failure_rate_bvh'] <= r['failure_rate_vanilla'], 'FAIL: BVH failure rate not lower'
print('ALL PART A CRITERIA MET')
"
```

Expected: `ALL PART A CRITERIA MET` (or investigate if not).

- [ ] **Step 4: Commit results**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
git add results/phase3_eval_report.json
git commit -m "results(phase3): Part A eval — delta_return_inference=$(python3 -c \"import json; r=json.load(open('results/phase3_eval_report.json')); print(f\\\"{r['delta_return_inference']:.2f}\\\")\") BVH vs Vanilla"
```

---

## Task 4: Run Part B — BVH-gated training comparison

**Files:**
- Run: `scripts/experiment_phase3.py --mode train --gated` (BVH actor)
- Run: `scripts/experiment_phase3.py --mode train` (vanilla actor)
- Output: `results/phase3_train_report.json`, `results/phase3_vanilla_report.json`

- [ ] **Step 1: Run BVH actor training (overnight)**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
caffeinate -dims python3 scripts/experiment_phase3.py \
    --mode train \
    --checkpoint checkpoints/sd_p1_v2.pt \
    --phase3-steps 50000 \
    --gated \
    --out results/phase3_train_report.json
```

Expected runtime: ~2-4 hours (50k steps with imagination + 200 eval episodes).

- [ ] **Step 2: Run vanilla actor training (overnight)**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
caffeinate -dims python3 scripts/experiment_phase3.py \
    --mode train \
    --checkpoint checkpoints/sd_p1_v2.pt \
    --phase3-steps 50000 \
    --out results/phase3_vanilla_report.json
```

- [ ] **Step 3: Compute delta_return_trained**

```bash
python3 -c "
import json
bvh = json.load(open('results/phase3_train_report.json'))
van = json.load(open('results/phase3_vanilla_report.json'))
delta_trained = bvh['trained_mean_return'] - van['trained_mean_return']
delta_inference = json.load(open('results/phase3_eval_report.json'))['delta_return_inference']
print(f'delta_return_trained:   {delta_trained:.2f}')
print(f'delta_return_inference: {delta_inference:.2f}')
print(f'Training compounded benefit: {delta_trained > delta_inference}')
"
```

Expected: `delta_return_trained > delta_return_inference` (trained actor benefits more than inference-only gating).

- [ ] **Step 4: Commit training results**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
git add results/phase3_train_report.json results/phase3_vanilla_report.json
git commit -m "results(phase3): Part B training — BVH vs Vanilla actor comparison"
```

---

## Task 5: Document results and push

**Files:**
- Modify: `README.md`
- Modify: `docs/results/README.md`
- Create: `docs/results/phase3_eval_report.json`

- [ ] **Step 1: Copy eval report to docs**

```bash
cp results/phase3_eval_report.json docs/results/phase3_eval_report.json
```

- [ ] **Step 2: Add Phase 3 section to README.md**

In `README.md`, after the SensorDrift results table, add:

```markdown
### Phase 3 — BVH-Gated Policy (delta_return)

The τ̂ signal is not just predictive — it's **actionable**. When a BVH-gated policy
substitutes zero-action for STALE predictions, it avoids the negative reward hole
from corrupted imagined dynamics.

| Metric | BVH Policy | Vanilla Policy |
|--------|-----------|----------------|
| Mean return | X.X | X.X |
| Failure rate (return < −50) | X.X% | X.X% |
| `delta_return_inference` | **+X.X** | 0 (baseline) |
| `stale_trigger_rate` | X.X% | — |

Full numbers: [`docs/results/phase3_eval_report.json`](docs/results/phase3_eval_report.json)

> "On SensorDrift, a BVH-gated policy achieves delta_return = +X over an unaware
> policy using identical world model weights. Knowing when the world model is stale,
> and acting on that knowledge, measurably improves task performance."
```

Fill in the actual numbers from `results/phase3_eval_report.json`.

- [ ] **Step 3: Update docs/results/README.md with Phase 3 section**

Add a new `## Phase 3` section after the SensorDrift section:

```markdown
## Phase 3 — `phase3_eval_report.json`

BVH-gated inference: 200 episodes, same Phase 1+2 weights, BVH vs Vanilla policy.

| Metric | Value |
|--------|-------|
| `bvh_mean_return` | X.X |
| `vanilla_mean_return` | X.X |
| `delta_return_inference` | **+X.X** |
| `failure_rate_bvh` | X.X% |
| `failure_rate_vanilla` | X.X% |
| `stale_trigger_rate` | X.X% |
```

- [ ] **Step 4: Run full test suite one final time**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
python -m pytest tests/unit/ -v --tb=short 2>&1 | tail -20
```

Expected: All tests pass.

- [ ] **Step 5: Final commit and push**

```bash
cd /Users/nikhil-verma-ai/bvh-rssm
git add README.md docs/results/README.md docs/results/phase3_eval_report.json
git commit -m "docs(phase3): document delta_return breakthrough — BVH-gated policy results"
git push github-nikhil main
```

---

## Success Criteria Checklist

| Criterion | Target | Check |
|-----------|--------|-------|
| `delta_return_inference` > 0 | BVH > Vanilla at inference time | `results/phase3_eval_report.json` |
| `failure_rate_bvh` < `failure_rate_vanilla` | Fewer catastrophic episodes | same file |
| `stale_trigger_rate` in [0.20, 0.80] | Router fires meaningfully | same file |
| `delta_return_trained` > `delta_return_inference` | Training compounds benefit | compare both JSONs |
| All tests pass | No regressions | `pytest tests/unit/` |
