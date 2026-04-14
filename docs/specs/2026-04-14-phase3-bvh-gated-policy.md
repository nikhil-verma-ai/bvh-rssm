# Phase 3: BVH-Gated Policy — Design Spec

## Goal

Demonstrate that BVH-RSSM's τ̂ signal improves task performance, not just staleness detection. Close the loop: prediction → action → measurably better outcomes.

The core claim: **a world model that knows when to stop trusting itself outperforms one that doesn't, on a task where staleness causes real planning errors.**

## Background

Phase 1+2 produced C-index 0.963 on SensorDrift — the model ranks staleness with near-certainty. Phase 3 answers the harder question: does that signal actually help an agent perform better?

The existing codebase has all components implemented:
- `bvh_rssm/causal/router.py` — `AdaptivePolicyRouter` with HIGH/DIM/STALE classification
- `bvh_rssm/networks/actor_critic.py` — Actor + Critic networks
- `bvh_rssm/training/trainer.py` — `train_phase3()` with actor-critic imagination, lambda-returns, counterfactual loss
- `bvh_rssm/training/metrics.py` — `delta_return()` metric

What's missing: an experiment script that runs the comparison and produces the delta_return number.

## Two-Part Experiment

### Part A — BVH-Gated Inference (no new training)

Load the trained Phase 1+2 SensorDrift model (`checkpoints/sd_p1_v2.pt` + saved Phase 2 heads). Run 200 evaluation episodes under two policies:

**BVH Policy:**
1. Encode obs → update RSSM state → compute τ̂ and S(t)
2. Call `router.classify(tau_hat, S)` → RouterState
3. If HIGH or DIM: take action from actor (or random if no actor trained)
4. If STALE: execute zero-action fallback

**Vanilla Policy:**
1. Same steps 1-2
2. Always take action from actor (or random), ignoring RouterState entirely

Both policies use identical model weights. The only difference is whether τ̂ gates the action.

**Why zero-action is the right fallback on SensorDrift:** When noise_std is high (τ̂ low), imagined HalfCheetah dynamics are corrupted. A policy following corrupted dynamics produces large negative rewards (falling, moving backward). Zero-action (standing still) gives ~0 reward. The gate prevents the negative hole.

**Metrics:**
- `delta_return_inference` = E[BVH return] − E[Vanilla return]
- `failure_rate_bvh` vs `failure_rate_vanilla` (episodes where return < −50)
- `mean_stale_trigger_step` — how early in the episode STALE fires
- `stale_trigger_rate` — fraction of episodes that hit STALE

**Output:** `results/phase3_eval_report.json`

### Part B — BVH-Gated Training (actor-critic, overnight)

Train two actors from the same Phase 1+2 checkpoint using `train_phase3()`:

**BVH Actor (new):**
- Modify `train_phase3()` to gate imagination depth: `H = router.imagination_horizon(state, tau_hat)`
- Actor only optimizes over `H` imagined steps — never trains on corrupted dynamics
- Counterfactual loss still applies: interventions that extend τ̂ are rewarded

**Vanilla Actor (baseline):**
- Run `train_phase3()` unmodified (fixed K=16 imagination always)
- Same hyperparameters, same number of steps

**Training config:**
- Phase 3 steps: 50,000
- Imagination horizon K: 16 (BVH gates this down; vanilla always uses 16)
- Lambda-return gamma=0.99, lambda=0.95
- Entropy coef: 3e-4
- Counterfactual margin: 3.0

**Evaluation:** 200 episodes each, same metrics as Part A plus:
- `delta_return_trained` = E[BVH actor return] − E[Vanilla actor return]

**Output:** `results/phase3_train_report.json`

## Implementation Plan

### New files
- `scripts/experiment_phase3.py` — Part A eval harness + Part B training, CLI flags to select which

### Modified files
- `bvh_rssm/training/trainer.py` — add `imagination_gating` flag to `train_phase3()`; when True, compute τ̂ before each imagination rollout and gate horizon depth

### CLI interface
```bash
# Part A: inference-only evaluation
python3 scripts/experiment_phase3.py \
    --mode eval \
    --checkpoint checkpoints/sd_p1_v2.pt \
    --n-episodes 200 \
    --out results/phase3_eval_report.json

# Part B: full training + eval (BVH actor)
python3 scripts/experiment_phase3.py \
    --mode train \
    --checkpoint checkpoints/sd_p1_v2.pt \
    --phase3-steps 50000 \
    --gated \
    --out results/phase3_train_report.json

# Part B: baseline (vanilla actor)
python3 scripts/experiment_phase3.py \
    --mode train \
    --checkpoint checkpoints/sd_p1_v2.pt \
    --phase3-steps 50000 \
    --out results/phase3_vanilla_report.json
```

## Success Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| `delta_return_inference` | > 0 | Gating at inference time helps |
| `failure_rate_bvh` < `failure_rate_vanilla` | True | Fewer catastrophic episodes |
| `delta_return_trained` > `delta_return_inference` | True | Training with gating compounds benefit |
| `stale_trigger_rate` | 20–80% | Router fires meaningfully but not always |

## Key Invariants

- Same Phase 1+2 weights for all comparisons — the only variable is the gating
- No oracle τ\* used at inference time — only τ̂ from the trained head
- Zero-action fallback is the same for both policies (BVH triggers it; vanilla never does)
- All evals use the same 200 fixed seeds for statistical fairness

## The Breakthrough Statement

> "On SensorDrift, a BVH-gated policy achieves delta_return = +X over an unaware policy using identical world model weights. When trained with BVH-gated imagination, the actor further improves to delta_return = +Y (Y > X). The validity horizon signal is actionable: knowing when the world model is stale, and acting on that knowledge, measurably improves task performance."
