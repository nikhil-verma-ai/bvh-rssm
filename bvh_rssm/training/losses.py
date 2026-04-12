"""
World model losses for BVH-RSSM (DreamerV3-style).

Four exported functions:
  kl_loss              — KL divergence between posterior and prior categoricals
  world_model_loss     — Full world model loss: prediction + dynamics + representation
  validity_loss        — Thin wrapper around ValidityHead.loss
  counterfactual_loss  — Hinge loss: penalize interventions that shorten validity horizon

KL convention (DreamerV3):
  - "dynamics" loss: KL(sg(post) || prior)  — trains the prior, gradients flow only to prior
  - "representation" loss: KL(post || sg(prior)) — trains the posterior, gradients flow only to post
  - Total KL = 0.8 * dynamics + 0.2 * representation
  - Free bits clamping at 1.0 nat per category (prevents KL collapse early in training)

Prediction loss (reconstruction):
  - NLL in symlog space: -log N(symlog(obs) | mean_symlog, std)
  - Numerically stable: operates entirely in symlog space, symexp never called here

References:
  Hafner et al. "Mastering Diverse Domains with World Models" (DreamerV3), 2023
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from bvh_rssm.utils import symlog, unimix


def kl_loss(
    posterior_logits: Tensor,
    prior_logits: Tensor,
    free_bits: float = 1.0,
    eps: float = 0.01,
) -> Tensor:
    """Categorical KL divergence KL(post || prior) with unimix and free bits.

    Applies unimix (eps mixture with uniform) to both distributions before
    computing KL. This prevents log(0) singularities when logits saturate.

    Free-bits clamping: per-category KL is clamped to [free_bits, inf) and
    then averaged. This prevents the model from collapsing to prior too early.

    Args:
        posterior_logits: [B, z_cats, z_classes] — raw posterior logits
        prior_logits:     [B, z_cats, z_classes] — raw prior logits
        free_bits: Minimum KL per category in nats (DreamerV3 default: 1.0).
        eps: Unimix uniform mixture coefficient (default: 0.01).

    Returns:
        Scalar mean KL loss >= 0.
    """
    # Apply unimix to both: mix softmax probs with uniform to prevent log(0)
    # Shape: [B, z_cats, z_classes]
    post_probs = unimix(posterior_logits, eps=eps)
    prior_probs = unimix(prior_logits, eps=eps)

    # KL(post || prior) per element, then sum over z_classes dimension
    # torch.distributions.kl handles the per-element computation cleanly but
    # we do it manually to avoid Categorical object construction overhead:
    #   KL(P||Q) = sum_x P(x) * log(P(x)/Q(x))
    # Shape after sum(-1): [B, z_cats]
    kl_per_cat = (post_probs * (post_probs.clamp(min=1e-8).log() - prior_probs.clamp(min=1e-8).log())).sum(-1)

    # DreamerV3: clamp per-category batch mean, then average over categories
    # kl_per_cat: [B, z_cats] → mean over batch → [z_cats] → clamp → scalar
    kl_per_cat_mean = kl_per_cat.mean(dim=0)          # [z_cats]
    return kl_per_cat_mean.clamp(min=free_bits).mean()


def world_model_loss(
    obs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    continues: Tensor,
    encoder,
    decoder,
    rssm,
    reward_head,
    continue_head,
    kl_dynamics_scale: float = 0.8,
    kl_repr_scale: float = 0.2,
    free_bits: float = 1.0,
    unimix_eps: float = 0.01,
) -> Dict[str, Tensor]:
    """Full DreamerV3-style world model loss over T timesteps.

    Unrolls the RSSM over an observed trajectory, computing:
      - prediction loss: NLL(obs | latent) + reward_loss + continue_loss
      - dynamics loss:   KL(sg(post) || prior) — trains prior
      - representation loss: KL(post || sg(prior)) — trains posterior

    Total = prediction + 0.8 * dynamics + 0.2 * representation

    Args:
        obs:           [B, T, obs_dim] — observed trajectory
        actions:       [B, T, action_dim] — actions taken at each step
        rewards:       [B, T] — rewards received
        continues:     [B, T] — episode continuation flags (1=continue, 0=done)
        encoder:       Encoder module — maps obs to embed
        decoder:       Decoder module — decode_symlog for NLL loss
        rssm:          RSSM module — observe() returns (post_logits, state)
        reward_head:   RewardHead — loss(latent, target) -> scalar
        continue_head: ContinueHead — loss(latent, continue_flag) -> scalar
        kl_dynamics_scale: Weight on dynamics KL term (default 0.8).
        kl_repr_scale: Weight on representation KL term (default 0.2).
        free_bits: Free-bits floor per category for KL (nats).
        unimix_eps: Uniform mixing coefficient for KL computation.

    Returns:
        Dict with keys: "total", "prediction", "dynamics", "representation"
        All values are scalar Tensors.
    """
    B, T, _ = obs.shape
    device = obs.device

    # Encode all observations in one batched pass: [B*T, obs_dim] -> [B*T, embed_dim]
    obs_flat = obs.reshape(B * T, -1)
    embed_flat = encoder(obs_flat)                           # [B*T, embed_dim]
    embeds = embed_flat.reshape(B, T, -1)                   # [B, T, embed_dim]

    # Initialize RSSM state
    state = rssm.initial_state(B, device=device)

    # Accumulators
    post_logits_list = []
    prior_logits_list = []
    latents_list = []

    # Unroll over time — observe uses action_{t-1} to step GRU before incorporating obs_t
    for t in range(T):
        post_logits, state = rssm.observe(
            embeds[:, t, :],    # obs embed at step t
            actions[:, t, :],   # action at step t (used as a_{t-1} for GRU step)
            state,
        )
        # Get prior for same h_t (prior doesn't use obs embed)
        prior_logits = rssm.prior_head(state.h)
        prior_logits = prior_logits.reshape(B, rssm.z_cats, rssm.z_classes)

        latent = rssm.get_latent(state)                     # [B, h_dim + z_dim]

        post_logits_list.append(post_logits)
        prior_logits_list.append(prior_logits)
        latents_list.append(latent)

    # Stack time dimension: [B, T, z_cats, z_classes]
    post_logits_all = torch.stack(post_logits_list, dim=1)
    prior_logits_all = torch.stack(prior_logits_list, dim=1)
    latents_all = torch.stack(latents_list, dim=1)          # [B, T, latent_dim]

    # Flatten B,T for head computations
    latents_flat = latents_all.reshape(B * T, -1)           # [B*T, latent_dim]

    # --- Prediction loss ---
    # Observation NLL in symlog space: -log N(symlog(obs) | mean_symlog, std)
    mean_symlog, log_std = decoder.decode_symlog(latents_flat)  # [B*T, obs_dim] each
    obs_symlog = symlog(obs_flat)                               # [B*T, obs_dim]
    std = log_std.exp()
    # NLL of isotropic Gaussian in symlog space, summed over obs_dim, meaned over B*T
    obs_nll = (
        0.5 * ((obs_symlog - mean_symlog) / std).pow(2)
        + log_std
        + 0.5 * math.log(2 * math.pi)
    ).sum(-1).mean()

    # Reward loss
    reward_flat = rewards.reshape(B * T)
    rew_loss = reward_head.loss(latents_flat, reward_flat)

    # Continue loss — ContinueHead.loss expects continue_flag [B, 1] per its signature
    # We pass [B*T, 1] which is the natural generalization
    cont_flat = continues.reshape(B * T, 1)
    cont_loss = continue_head.loss(latents_flat, cont_flat)

    pred_loss = obs_nll + rew_loss + cont_loss

    # --- KL losses ---
    # Flatten B,T into batch dimension: [B*T, z_cats, z_classes]
    post_flat = post_logits_all.reshape(B * T, rssm.z_cats, rssm.z_classes)
    prior_flat = prior_logits_all.reshape(B * T, rssm.z_cats, rssm.z_classes)

    # Dynamics loss: KL(sg(post) || prior) — stop grad on posterior, train prior
    dyn_loss = kl_loss(post_flat.detach(), prior_flat, free_bits=free_bits, eps=unimix_eps)

    # Representation loss: KL(post || sg(prior)) — stop grad on prior, train posterior
    repr_loss = kl_loss(post_flat, prior_flat.detach(), free_bits=free_bits, eps=unimix_eps)

    total = pred_loss + kl_dynamics_scale * dyn_loss + kl_repr_scale * repr_loss

    return {
        "total": total,
        "prediction": pred_loss,
        "dynamics": dyn_loss,
        "representation": repr_loss,
    }


def validity_loss(
    tau_head,
    latent: Tensor,
    action: Tensor,
    oracle_tau: Tensor,
    stop_grad: bool = True,
) -> Tensor:
    """Thin wrapper around ValidityHead.loss.

    Delegates directly to the head's loss method — no additional logic needed
    since ValidityHead handles stop_grad internally via latent.detach().

    Args:
        tau_head:   ValidityHead instance.
        latent:     [B, latent_dim] — concatenated RSSM latent.
        action:     [B, action_dim] — action at this step.
        oracle_tau: [B] — ground-truth steps-to-distribution-shift.
        stop_grad:  If True, detach latent before processing (Phase 2 training).

    Returns:
        Scalar loss.
    """
    return tau_head.loss(latent, action, oracle_tau, stop_grad=stop_grad)


def counterfactual_loss(
    tau_int: Tensor,
    tau_obs: Tensor,
    margin: float = 3.0,
) -> Tensor:
    """Counterfactual hinge loss: penalize interventions that fail to improve horizon.

    An intervention is useful if it extends the validity horizon beyond the
    observational baseline. This loss is zero when tau_int >= tau_obs + margin
    (intervention clearly helps), and positive otherwise.

    Formula: mean(relu(tau_obs - tau_int + margin))

    Invariants:
      - loss >= 0 always (relu)
      - loss = 0 when tau_int >= tau_obs + margin for all batch elements

    Args:
        tau_int:  [B] — validity horizon under intervention action.
        tau_obs:  [B] — validity horizon under observed/baseline action.
        margin:   Minimum required improvement in steps (default: 3.0).

    Returns:
        Scalar mean hinge loss.
    """
    return F.relu(tau_obs - tau_int + margin).mean()


def survival_loss(
    hazard_head,
    latent: Tensor,
    event_times: Tensor,
    event_occurred: Tensor,
    use_all_sources: bool = False,
) -> Tensor:
    """Proper discrete-time survival negative log-likelihood (Discrete-time Cox NLL).

    For each sample b with event interval t_b and observed/censored flag d_b:
      - Observed (d_b=True):  log L_b = log h(t_b) + sum_{i<t_b} log(1 - h(i))
      - Censored (d_b=False): log L_b = sum_{i<=t_b} log(1 - h(i))

    This is the exact discrete-time likelihood from Tutz & Schmid (2016), not
    the BCE approximation in loss_source_b(). The cumsum trick vectorizes the
    sequential summation over intervals without a Python loop.

    Clamp hazards to [1e-7, 1-1e-7] before log to prevent NaN from log(0).

    Args:
        hazard_head:    HazardHead instance.
        latent:         [B, latent_dim] — RSSM latent at this step.
        event_times:    [B] integer interval indices (0-indexed). Clipped to [0, K-1].
        event_occurred: [B] bool, True=observed event, False=right-censored.
        use_all_sources: If True, use combined_hazard (all three sources A+B+C).
                         If False, use source B only (Phase 2 training mode).

    Returns:
        Scalar NLL >= 0.

    Complexity: O(B * K). All ops are vectorized — no Python loop over batch or time.
    Side effects: None.
    """
    if use_all_sources:
        h = hazard_head.combined_hazard(latent)       # [B, K]
    else:
        _, h, _ = hazard_head.forward(latent)         # [B, K]

    B, K = h.shape
    device = latent.device

    # Clamp before taking logs to prevent log(0) → -inf → NaN propagation
    h_clamped = h.clamp(1e-7, 1.0 - 1e-7)            # [B, K]
    log_h = torch.log(h_clamped)                      # [B, K]: log hazard at each interval
    log_surv = torch.log(1.0 - h_clamped)             # [B, K]: log survival increment

    # Build cumulative log survival: cs[b, t] = sum_{i=0}^{t-1} log(1 - h[b,i])
    # So cs[:, 0] = 0 (no time elapsed), cs[:, 1] = log_surv[:,0], etc.
    # Shape: [B, K+1] — the +1 column lets us index cs_at_t1 = cs[:, t+1] without OOB.
    cs = torch.zeros(B, K + 1, device=device, dtype=h.dtype)
    cs[:, 1:] = torch.cumsum(log_surv, dim=-1)        # [B, K] cumsum → slots 1..K

    arange_b = torch.arange(B, device=device)
    t = event_times.long().clamp(0, K - 1)            # [B] — clip invalid indices
    obs_mask = event_occurred.bool()                   # [B]

    # Gather log-hazard at the event interval for observed samples
    log_h_at_t = log_h[arange_b, t]                   # [B]

    # cs[b, t_b]   = cumulative log-survival up to (not including) t_b
    # cs[b, t_b+1] = cumulative log-survival up to (including) t_b
    cs_at_t = cs[arange_b, t]                         # [B]
    cs_at_t1 = cs[arange_b, t + 1]                   # [B] — safe since cs has K+1 cols

    # Observed: log L_b = log h(t_b) + cs[b, t_b]
    # Censored: log L_b = cs[b, t_b+1]
    log_lik = torch.where(obs_mask, log_h_at_t + cs_at_t, cs_at_t1)  # [B]

    # Return negative mean log-likelihood (minimization objective)
    return -log_lik.mean()
