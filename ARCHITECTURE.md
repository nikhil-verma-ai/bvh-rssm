# BVH-RSSM Architecture

Technical reference for contributors and reviewers.

---

## Module Dependency Graph

```
bvh_rssm/
├── networks/          # Pure torch.nn — no training deps
│   ├── common.py      # MLP, LayerNormMLP building blocks
│   ├── encoder.py     # Obs → embed (symlog + LayerNormMLP)
│   ├── decoder.py     # Latent → obs distribution (symlog space)
│   ├── rssm.py        # GRU + categorical z_t (State namedtuple)
│   ├── heads.py       # RewardHead, ContinueHead, ValidityHead, HazardHead
│   └── actor_critic.py# Actor (policy), Critic (value, twohot bins)
│
├── utils/             # Math primitives — no torch.nn
│   ├── math.py        # symlog, symexp, symlog_bins, twohot, twohot_decode, unimix
│   ├── distributions.py# straight_through_sample, sample_categorical (STE)
│   └── rng.py         # RNGStateStore, save/restore_rng_state, rng_snapshot
│
├── training/          # Requires training optional deps (wandb, hydra, pycox)
│   ├── losses.py      # kl_loss, world_model_loss, validity_loss, survival_loss
│   ├── replay_buffer.py# Circular buffer (obs, action, reward, oracle_tau, rng_state)
│   ├── trainer.py     # 3-phase Trainer + TrainerConfig
│   ├── collector.py   # EnvStepper — online env→buffer transition collection
│   └── experiment.py  # set_seed, init_wandb, Checkpointer, log_metrics
│
├── envs/              # Gymnasium wrappers — ShiftWrapper base class
│   ├── wrappers.py    # ShiftWrapper: periodic shift scheduling, oracle_tau in info
│   ├── shift_pendulum.py, shift_walker.py, shift_maze.py
│   ├── regime_maze.py, trading_regime.py, sensor_drift.py
│
├── causal/            # BVH router and attribution
│   ├── router.py      # BVH Router: HIGH/DIM/STALE classification
│   └── attribution.py # Pearl's 3 levels, counterfactual rollout
│
└── serving/           # FastAPI inference server
    ├── predictor.py   # Stateless Predictor — loads model, runs single-step inference
    ├── server.py      # FastAPI app factory (create_app)
    └── schemas.py     # Pydantic request/response schemas
```

**Dependency rule:** `networks/` and `utils/` import only PyTorch and each other.
`training/` imports `networks/`, `utils/`, and optional training deps.
`envs/` imports only NumPy and Gymnasium. Nothing imports from `training/`.

---

## RSSM Internals

The RSSM maintains a two-part state `(h_t, z_t)` at each timestep:

```
h_t = GRU(LayerNorm([z_{t-1}, a_{t-1}]), h_{t-1})    # deterministic recurrent
z_t ~ Categorical(q(z_t | h_t, embed_t))               # stochastic posterior
```

**GRU input normalization:** `[z_{t-1}, a_{t-1}]` is LayerNorm'd before the GRU cell.
This stabilizes gradients through the long recurrence — without it, KL collapse
(posterior → prior) occurs within 5k steps.

**Categorical z_t (32×32):** 32 independent categorical distributions, each with 32
classes. The one-hot samples are concatenated to form a 1024-dim `z_t`. This factored
structure lets each of the 32 groups learn an independent "concept" (regime indicator,
velocity component, etc.).

**Straight-through estimator:** Sampling is non-differentiable. STE routes the identity
gradient through the one-hot operation: backward pass receives the dense gradient as if
the argmax were a differentiable operation. Implementation: custom `autograd.Function`
in `utils/distributions.py`. The naive `one_hot + probs - probs.detach()` formulation
vanishes for sum-invariant losses — the custom function avoids this.

**Unimix (ε=0.01):** Before computing KL(post||prior), both distributions are mixed with
a uniform: `p_mix = (1-ε)·softmax(logits) + ε/n_classes`. This prevents `log(0)`
singularities when logits saturate early in training. Cost: KL lower bound is tightened
by `ε·log(n_classes)` ≈ 0.047 nat per category.

**Free-bits clamping:** KL per category is clamped at 1.0 nat minimum before averaging.
This prevents the RSSM from ignoring low-information categories early in training.

---

## Three-Phase Training Pipeline

### Phase 1 — World Model Pretraining

Trains: `encoder`, `decoder`, `rssm`, `reward_head`, `continue_head`, `tau_head` (joint).
Freezes: `hazard_head`.

Loss:
```
L_P1 = L_pred + 0.8·L_dyn + 0.2·L_repr + 0.3·L_τ_aux
```

- `L_pred` = NLL(symlog(obs) | decoder_mean, std) + twohot CE(reward) + BCE(continue)
- `L_dyn` = KL(sg(post) || prior) — trains prior
- `L_repr` = KL(post || sg(prior)) — trains posterior
- `L_τ_aux` = twohot CE(τ̂ | oracle_τ\*) with `stop_grad=False` — shapes latent

The joint τ auxiliary loss (weight 0.3) trains the encoder/RSSM to preserve τ-relevant
information in `h_t`. Without it, Phase 2 head training supervises regression on features
with no τ signal, making convergence very slow.

### Phase 2 — BVH Head Training

Trains: `tau_head`, `hazard_head`.
Freezes: all world model weights.

Loss:
```
L_P2 = L_τ + L_survival
```

- `L_τ` = twohot CE with `stop_grad=True` (latent detached — pure head training)
- `L_survival` = discrete-time Cox NLL (proper survival likelihood, not BCE approx)

**Stop-grad invariant:** The KL divergence of the world model must not change by more
than 0.5 nat during Phase 2. Verified: v2 run shows ΔKL = −0.001 nat. This confirms
head training is not leaking gradients into the frozen RSSM (the `latent.detach()`
in `ValidityHead.forward` and `HazardHead.forward` is effective).

**Burn-in:** For online collection sequences of length 64, the first 32 timesteps are
excluded from the τ loss. The GRU `h_t` needs time to accumulate evidence of a regime
shift before being asked to predict τ\*.

### Phase 3 — Joint Fine-Tuning + Actor-Critic (Optional)

Trains all weights jointly. Actor-critic in imagination rollouts.

Loss:
```
L_P3 = L_wm + L_τ(stop_grad=False) + L_survival + L_cf + L_actor + L_critic
```

- `L_cf` = counterfactual hinge: `relu(τ_obs - τ_int + margin)` — reward interventions
  that extend the validity horizon
- `L_actor` = -E[R^λ] + entropy_coef·H[π]
- `L_critic` = twohot CE against λ-returns (stop-grad)

---

## BVH Heads

### ValidityHead (τ-head)

Input: `[h_t; z_t]` (latent_dim) + action (action_dim → 64-dim embedding)
Output: logits over n_bins twohot bins in symlog([0, 1000]) space

```
τ̂ = symexp(Σ_i p_i · bin_i)    where p_i = softmax(logits)_i
```

Training loss: `L_τ = -Σ_i twohot(symlog(τ*))_i · log_softmax(logits)_i`

The twohot encoding places soft mass on the two bins bracketing `symlog(τ*)`, with
weights proportional to fractional distance. This provides smooth gradients near the
true value and handles the wide dynamic range of τ* (0 to 1000 steps).

### HazardHead (λ-head)

Three competing-risk sub-heads (Sources A, B, C), each outputting K=16 hazard
probabilities `h_{source}(i) ∈ (0, 1)` for each interval i.

```
h_total(i) = 1 - ∏_{X∈{A,B,C}} (1 - h_X(i))
S(t)        = ∏_{i<t} (1 - h_total(i))
```

Phase 1/2: Only Source B is active. Sources A and C are zero-initialized
(`bias=-5.0` → `sigmoid(-5) ≈ 0.007` contribution). They are activated in Phase 3
for multi-source causal attribution.

Training loss (proper discrete-time Cox NLL):
- Observed (shift at t): `log L = log h(t) + Σ_{i<t} log(1-h(i))`
- Censored (no shift in K steps): `log L = Σ_{i≤t} log(1-h(i))`

The cumsum trick vectorizes the sequential summation without a Python loop: O(B·K)
all-vectorized.

---

## Key Invariants

These are checked after every full validation run (`scripts/validate.py`):

1. **KL not collapsed:** `kl_after_p1 > 0.1 nat` — RSSM posterior is not just prior
2. **Stop-grad:** `|kl_after_p2 - kl_before_p2| < 0.5 nat` — heads don't break WM
3. **τ beats naive mean:** `mae_tau < naive_mean_mae` — heads learn something
4. **τ beats naive zero:** `mae_tau < naive_zero_mae` — basic sanity check
5. **Loss decreasing:** Phase 1 loss must decrease from early to late checkpoint
6. **Prediction not collapsed:** `pred_std > 0.5` — τ̂ has non-trivial variance

---

## Inference Serving

`scripts/serve.py` launches a FastAPI server exposing a single endpoint:

```
POST /predict
{
  "obs": [float, ...],      # raw observation vector
  "action": [float, ...],   # action vector
  "state": null | {...}     # RSSM state (null = episode start)
}
```

Response:
```json
{
  "tau_hat": 42.3,          # predicted steps until world model staleness
  "survival": [0.9, 0.7, ...], # S(t) for t=0..K-1
  "router_state": "HIGH",   # HIGH / DIM / STALE
  "state": {...}             # updated RSSM state for next call
}
```

The `Predictor` class is stateless between calls — the `state` dict encodes the full
RSSM state (h, z as lists of floats) and is round-tripped through JSON. This enables
horizontal scaling without server-side session storage.
