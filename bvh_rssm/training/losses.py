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
