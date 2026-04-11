"""
RSSM — Recurrent State Space Model (DreamerV3 implementation from scratch).

Architecture:
  h_t = GRU(LayerNorm([z_{t-1}, a_{t-1}]), h_{t-1}) — deterministic recurrent state
  z_t ~ Categorical(posterior_logits)                   — stochastic categorical state

Two forward modes:
  observe(obs_embed, action, state)  — posterior: uses real observation o_t
  imagine(action, state)              — prior: imagination-only, no observation needed

The full latent is cat([h_t, z_t]) used as input to all heads.

Key design decisions:
  - Unimix ε=0.01 applied to all categorical distributions (prevents log(0))
  - Straight-through gradient through discrete z_t sampling
  - z_t shape: [B, z_cats, z_classes] → flattened to [B, z_cats*z_classes]
  - LayerNorm on GRU input for training stability
"""
from __future__ import annotations

from typing import Tuple
from collections import namedtuple

import torch
import torch.nn as nn
from torch import Tensor

from bvh_rssm.networks.common import MLP
from bvh_rssm.utils import unimix, straight_through_sample, sample_categorical

# State namedtuple used everywhere in the codebase
State = namedtuple("State", ["h", "z"])


class RSSM(nn.Module):
    """Recurrent State Space Model.

    Args:
        h_dim: GRU hidden state dimension.
        z_cats: Number of categorical distributions (stochastic latent groups).
        z_classes: Classes per categorical.
        obs_dim: Observation embedding dimension (encoder output dim).
        action_dim: Action space dimension (continuous or embedded discrete).
        unimix_eps: Mixing coefficient for uniform distribution (prevents log(0)).
    """

    def __init__(
        self,
        h_dim: int = 512,
        z_cats: int = 32,
        z_classes: int = 32,
        obs_dim: int = 1024,
        action_dim: int = 6,
        unimix_eps: float = 0.01,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.z_cats = z_cats
        self.z_classes = z_classes
        self.z_dim = z_cats * z_classes
        self.unimix_eps = unimix_eps

        # GRU input: cat([z_{t-1}, a_{t-1}])
        gru_input_dim = self.z_dim + action_dim
        self.gru_cell = nn.GRUCell(input_size=gru_input_dim, hidden_size=h_dim)
        self.gru_norm = nn.LayerNorm(gru_input_dim)

        # Prior: p(z_t | h_t) — imagination-only
        self.prior_head = MLP(
            in_dim=h_dim,
            out_dim=z_cats * z_classes,
            hidden_dim=h_dim,
            n_layers=1,
        )

        # Posterior: q(z_t | h_t, obs_embed) — uses real observation
        self.posterior_head = MLP(
            in_dim=h_dim + obs_dim,
            out_dim=z_cats * z_classes,
            hidden_dim=h_dim,
            n_layers=1,
        )

    def initial_state(self, batch_size: int, device: torch.device = None) -> State:
        """Return zero-initialized RSSM state for the start of an episode.

        Args:
            batch_size: Number of parallel environments.
            device: Target device. Infers from model parameters if None.

        Returns:
            State with h=[B, h_dim] zeros and z=[B, z_dim] zeros.
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return State(
            h=torch.zeros(batch_size, self.h_dim, device=device),
            z=torch.zeros(batch_size, self.z_dim, device=device),
        )

    def _gru_step(self, z_prev: Tensor, action: Tensor, h_prev: Tensor) -> Tensor:
        """One GRU step: h_t = GRU(LayerNorm([z_{t-1}, a_{t-1}]), h_{t-1})."""
        gru_input = torch.cat([z_prev, action], dim=-1)  # [B, z_dim + action_dim]
        gru_input = self.gru_norm(gru_input)
        return self.gru_cell(gru_input, h_prev)           # [B, h_dim]

    def _sample_z(self, logits: Tensor) -> Tensor:
        """Sample z_t from categorical logits with straight-through gradient.

        During training: stochastic sampling (multinomial) for posterior diversity.
        During eval: deterministic argmax for reproducible inference.
        Both paths use straight-through gradient through the one-hot sample.

        Unimix (eps=0.01) is applied to the distribution before sampling to
        prevent log(0) in downstream KL computation.

        Args:
            logits: Shape [B, z_cats, z_classes] — raw (pre-softmax) logits.

        Returns:
            Flat one-hot z_t of shape [B, z_cats * z_classes].
        """
        # Apply unimix: mix softmax probs with uniform to prevent log(0) in KL
        mixed_probs = unimix(logits, eps=self.unimix_eps)
        # Convert mixed probs back to log-space for sampling
        logits_for_sample = torch.log(mixed_probs.clamp(min=1e-8))
        if self.training:
            # Stochastic: explore categorical space during training
            z = sample_categorical(logits_for_sample)
        else:
            # Deterministic: MAP estimate at eval/inference time
            z = straight_through_sample(logits_for_sample)
        return z.reshape(z.shape[0], -1)                  # [B, z_dim]

    def observe(
        self, obs_embed: Tensor, action: Tensor, state: State
    ) -> Tuple[Tensor, State]:
        """Posterior update: incorporate real observation.

        Computes h_t from GRU, then z_t from q(z_t | h_t, obs_embed).

        Args:
            obs_embed: Encoded observation of shape [B, obs_dim].
            action: Action taken at t-1, shape [B, action_dim].
            state: Previous State(h, z).

        Returns:
            (posterior_logits, next_state) where:
              posterior_logits: [B, z_cats, z_classes] — for KL computation
              next_state: State with updated h and z
        """
        h = self._gru_step(state.z, action, state.h)         # [B, h_dim]
        posterior_input = torch.cat([h, obs_embed], dim=-1)   # [B, h_dim + obs_dim]
        posterior_logits = self.posterior_head(posterior_input)
        posterior_logits = posterior_logits.reshape(
            h.shape[0], self.z_cats, self.z_classes
        )                                                      # [B, z_cats, z_classes]
        z = self._sample_z(posterior_logits)                  # [B, z_dim]
        return posterior_logits, State(h=h, z=z)

    def imagine(
        self, action: Tensor, state: State
    ) -> Tuple[Tensor, State]:
        """Prior update: imagination without real observation.

        Computes h_t from GRU, then z_t from p(z_t | h_t).

        Args:
            action: Action at t-1, shape [B, action_dim].
            state: Previous State(h, z).

        Returns:
            (prior_logits, next_state) where:
              prior_logits: [B, z_cats, z_classes]
              next_state: State with updated h and z
        """
        h = self._gru_step(state.z, action, state.h)         # [B, h_dim]
        prior_logits = self.prior_head(h)
        prior_logits = prior_logits.reshape(
            h.shape[0], self.z_cats, self.z_classes
        )                                                      # [B, z_cats, z_classes]
        z = self._sample_z(prior_logits)                      # [B, z_dim]
        return prior_logits, State(h=h, z=z)

    def get_latent(self, state: State) -> Tensor:
        """Concatenate h and z into the full latent for head inputs.

        Args:
            state: Current State(h, z).

        Returns:
            Concatenated latent of shape [B, h_dim + z_dim].
        """
        return torch.cat([state.h, state.z], dim=-1)
