# BVH-RSSM — Belief Validity Horizon World Model

[![CI](https://github.com/vanshverma/bvh-rssm/actions/workflows/ci.yml/badge.svg)](https://github.com/vanshverma/bvh-rssm/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A world model that signals its own staleness — before failure, not after.**

Standard model-based RL agents fail silently: when their world model's internalized
dynamics go stale (after a distribution shift), predictions degrade but the agent
has no mechanism to detect this. In autonomous vehicle planning, a stale world model
that doesn't know it's stale is a safety liability.

**BVH-RSSM** extends [DreamerV3](https://arxiv.org/abs/2301.04104) with two novel
predictive heads that give the world model explicit self-awareness about its own
validity horizon:

- **τ-head (ValidityHead):** Predicts τ\* — the number of steps until the imagined
  latent trajectory diverges from the posterior by more than ε nats. Trained with
  twohot cross-entropy against oracle τ\* labels from the environment.
- **λ-head (HazardHead):** Models the distribution over shift times as a discrete-time
  survival process with competing risks (A, B, C). Trained with the proper discrete-time
  Cox negative log-likelihood.

---

## Results

Validation on **ShiftPendulum** (gravity-shift environment, Poisson shift process).
Full numbers: [`docs/results/v2_validation.json`](docs/results/v2_validation.json).

| Metric | BVH-RSSM | Naive Mean | Naive Zero |
|--------|----------|------------|------------|
| MAE (steps) | **7.24** | 7.60 | 10.94 |
| Beats naive mean | ✓ | — | — |

| Invariant Check | Result |
|----------------|--------|
| P1 loss decreasing | ✓ |
| KL not collapsed (>0.1 nat) | ✓ |
| Stop-grad invariant (Δ<0.5 nat) | ✓ |
| τ loss decreasing | ✓ |
| Beats naive mean MAE | ✓ |
| Beats naive zero MAE | ✓ |
| **6/7 checks passing** | |

> **C-index note:** ShiftPendulum uses a memoryless Poisson shift process (exponential
> inter-arrival times), making future shift times theoretically unpredictable from
> instantaneous observations. C-index > 0.65 is expected on SensorDrift-v0, where the
> oracle τ\* is deterministically encoded in the growing observation noise level.

---

## Architecture

```
Observation o_t ──► Encoder ──► embed_t ──┐
Action a_{t-1} ──────────────────────────►│
                                           ▼
                              ┌─────────────────────┐
h_{t-1} ──────────────────── │   RSSM (GRU + z_t)  │ ──► h_t, z_t
                              └─────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                       ▼
             τ-head (validity)      λ-head (hazard)         Decoder/Reward
             → τ* steps until       → S(t) survival         → p(o_t), r_t
               world model stale      function
                    │
                    ▼
             BVH Router
             HIGH / DIM / STALE
```

**Core components:**

| Module | Description |
|--------|-------------|
| `RSSM` | GRU recurrent state + categorical `z_t` (32×32 = 1024 classes). Straight-through gradient, unimix ε=0.01 |
| `Encoder` | Observation → embedding via symlog-compressed LayerNormMLP |
| `Decoder` | Latent → Gaussian in symlog space. NLL reconstruction loss |
| `ValidityHead` | [h,z]+action → τ̂ via twohot bins over [0, symlog(1000)] |
| `HazardHead` | [h,z] → h_A, h_B, h_C (competing risks) → survival S(t) |
| `BVH Router` | Classifies (HIGH/DIM/STALE) from τ̂ and S(t) thresholds |

Full technical reference: [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## Quick Start

Requires Python 3.11+.

```bash
git clone https://github.com/vanshverma/bvh-rssm.git
cd bvh-rssm
pip install -e ".[training]"  # core + training deps (no MuJoCo required for ShiftPendulum)
```

**Smoke test (fast mode, ~10 seconds):**
```bash
python3 scripts/train.py --fast
```

Expected output:
```
[train] device=mps  fast_mode=True  seed=42
[Phase 0] Collecting 500 random steps …
[Phase 0] Done — 500 steps in buffer (500 total)
[Phase 1] World model pretraining for 100 steps …
[Phase 1] Done
[Phase 2] BVH head training for 50 steps …
[Phase 2] Done
[train] All phases complete.
```

**Reproduce validation (uses saved P1 checkpoint, ~25 minutes):**
```bash
python scripts/validate.py --skip-p1
```

**Full validation with fresh P1 training (~2 hours):**
```bash
python scripts/validate.py --save-p1 checkpoints/my_p1.pt
```

---

## Environments — FNSB Benchmark

Six non-stationary environments with ground-truth oracle τ\* labels:

| Environment | Shift Type | Oracle τ | Obs Dim | Action Dim | Deps |
|-------------|-----------|----------|---------|------------|------|
| `ShiftPendulum` | Gravity (abrupt/gradual) | Poisson | 3 | 1 | None |
| `ShiftWalker` | Friction | Poisson | 17 | 6 | MuJoCo |
| `ShiftMaze` | Wall permeability | Abrupt | 7 | 3 | MiniGrid |
| `RegimeMaze` | Switch-button | Abrupt | 7 | 3 | MiniGrid |
| `TradingRegime` | HMM regime | Markov | 5 | 3 | None |
| `SensorDrift` | Monotonic noise | **Deterministic** | 17 | 6 | MuJoCo |

`SensorDrift` has the most predictable oracle τ (linear noise growth → exact τ\* from noise level), making it ideal for C-index evaluation.

---

## Design Decisions

**Why categorical latents (z_t)?**
Categorical distributions have bounded entropy and natural multi-modality — better for
discrete regime transitions than Gaussian. DreamerV3's 32×32 categories (1024 classes
total) provide enough capacity while keeping the KL tractable with free-bits clamping.

**Why twohot encoding for τ̂?**
τ\* spans 0–1000 steps, crossing multiple orders of magnitude. Twohot in symlog space
(per DreamerV3's reward head design) decouples gradient magnitude from prediction scale
and enables sharp probability mass near the true value without saturating gradients.

**Why survival analysis for λ-head?**
The shift arrival time is a random variable with censored observations (τ\* > K when
no shift occurs in K steps). Survival analysis handles censoring correctly via the
discrete-time Cox NLL — fitting a point estimator to censored data would introduce
systematic bias toward small τ values.

**Why stop-grad in Phase 2?**
The KL invariant (Δ < 0.5 nat change in KL during head training) verifies that the
head losses are not destroying the world model's latent structure. Validated empirically:
KL change = −0.001 nat in v2 run.

**Why unimix (ε=0.01)?**
Prevents log(0) singularities when posterior/prior categorical logits saturate during
early training. Applied to both distributions before KL computation — costs ~1% entropy
bound tightness for complete numerical stability.

**Why joint τ training in Phase 1?**
Phase 1 world model training with an auxiliary τ loss (weight=0.3) shapes the latent
encoder to preserve τ-relevant information. Without this, the RSSM has no incentive
to encode "how long ago did I last see a regime shift" in `h_t`, making Phase 2 head
training effectively supervised regression on random features.

---

## Reproduce

```bash
# Full validation with P1 checkpoint (P2 only, ~25 min)
python scripts/validate.py --skip-p1

# Full end-to-end (P1 + P2, ~2 hours, saves checkpoint)
python scripts/validate.py --save-p1 checkpoints/run.pt

# Fast smoke test (~10s)
python3 scripts/train.py --fast
```

Results are written to `validation_report.json`. Reference results: [`docs/results/`](docs/results/).

---

## Citation

If you use BVH-RSSM or the FNSB benchmark environments in your research:

```bibtex
@software{bvh_rssm_2026,
  author    = {Vansh Verma},
  title     = {{BVH-RSSM}: Belief Validity Horizon Recurrent State Space Model},
  year      = {2026},
  url       = {https://github.com/vanshverma/bvh-rssm},
  note      = {Research implementation}
}
```
