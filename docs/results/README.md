# BVH-RSSM Validation Results

Results from `python scripts/validate.py` on ShiftPendulum-v0 (gravity-shift environment).
Run: v2, 100k P1 steps (joint τ training) + 50k P2 steps, seed=42, device=mps.

## v2 Validation — `v2_validation.json`

### Phase 1 — World Model Pretraining

| Metric | Value | Meaning |
|--------|-------|---------|
| `loss_early` | 7.69 | World model loss at step 0 |
| `loss_late` | 1.89 | World model loss at step 100k |
| `kl_after` | 0.533 nat | KL divergence post-training (>0.1 = not collapsed) |

**Pass:** Loss decreased 7.69 → 1.89. KL healthy (not collapsed to prior).

### Phase 2 — BVH Head Training

| Metric | Value | Meaning |
|--------|-------|---------|
| `tau_loss_early` | 3.52 | τ-head cross-entropy at start of P2 |
| `tau_loss_late` | 3.48 | τ-head cross-entropy at end of P2 |
| `kl_increase` | −0.001 nat | Change in world-model KL during P2 |

**Stop-grad invariant:** KL changed by only −0.001 nat during head training (threshold: <0.5 nat).
This confirms head training did not perturb the world model's latent representation.

### Evaluation — 2000 samples across 50 episodes

| Metric | Value | Meaning |
|--------|-------|---------|
| `mae_tau` | 7.24 | Mean absolute error of τ̂ prediction (steps) |
| `naive_mean_mae` | 7.60 | MAE if always predicting the dataset mean |
| `naive_zero_mae` | 10.94 | MAE if always predicting τ=0 |
| `c_index` | 0.504 | Concordance index (0.5=random, 1.0=perfect ranking) |
| `brier_score` | 0.200 | Survival probability calibration |

### Pass/Fail Summary

| Check | Result |
|--------|--------|
| P1 loss decreasing | ✓ PASS |
| KL not collapsed | ✓ PASS |
| Stop-grad invariant | ✓ PASS |
| τ loss decreasing | ✓ PASS |
| Beats naive mean MAE | ✓ PASS |
| Beats naive zero MAE | ✓ PASS |
| Prediction not collapsed | ✗ FAIL (pred_std=0.31, threshold >0.5) |

**6/7 checks passing.**

**Note:** C-index (0.504) is reported as an informational metric, not a gating check. ShiftPendulum's Poisson shift process makes ranking theoretically difficult at chance; see C-index Note below.

### C-index Note

C-index of 0.50 on ShiftPendulum is expected: the Poisson shift process is memoryless
(exponential inter-arrival times), meaning future shift times are theoretically
unpredictable from the current observation alone. The RSSM can detect regime changes
via prediction-error spikes in `h_t`, but this requires multi-step temporal context
that 16-step training sequences do not fully provide.

**Expected improvement:** SensorDrift-v0 (monotonic sensor noise, deterministic oracle τ)
should yield C-index > 0.65 because τ* is directly encoded in observation magnitude.
Requires MuJoCo installation (`pip install -e ".[training]"`).
