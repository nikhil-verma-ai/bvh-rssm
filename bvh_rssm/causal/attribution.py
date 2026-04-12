"""
CausalAttributor — Pearl three-level hierarchy for BVH-RSSM.

Levels:
  1. Associational: P(τ | latent, action) — standard forward pass through tau_head.
  2. Interventional: do(a=alt_action) — imagine next state with alt_action,
     query tau_head on resulting latent.
  3. Counterfactual: same z_t as factual, different action — restore RNG to
     the state captured before the factual imagine call, re-run imagine with
     alt_action so the stochastic z_t sample is identical.

Invariants:
  - No trainable parameters. All three methods run under torch.no_grad()
    implicitly (the caller controls grad context if needed).
  - rssm_state is never mutated. imagine() returns a new State namedtuple.
  - counterfactual() restores the global RNG to rng_state before the imagine call,
    pinning z_t to its factual value (abduction of exogenous noise u_z).
"""
from __future__ import annotations

import torch
from torch import Tensor

from bvh_rssm.networks.rssm import RSSM, State
from bvh_rssm.networks.encoder import Encoder
from bvh_rssm.networks.heads import ValidityHead
from bvh_rssm.utils.rng import restore_rng_states


class CausalAttributor:
    """Implements the Pearl causal hierarchy for validity-horizon attribution.

    Wraps a frozen RSSM, Encoder, and ValidityHead. Holds no parameters.

    Args:
        rssm: Recurrent State Space Model (already trained / loaded).
        encoder: Observation encoder (already trained / loaded).
        tau_head: ValidityHead that maps (latent, action) → τ̂.
    """

    def __init__(self, rssm: RSSM, encoder: Encoder, tau_head: ValidityHead) -> None:
        self.rssm = rssm
        self.encoder = encoder
        self.tau_head = tau_head

    def associational(self, latent: Tensor, action: Tensor) -> Tensor:
        """Level 1: P(τ | latent, action) — standard forward pass.

        Passes the pre-computed latent directly through tau_head without any
        RSSM step. Use this when you have already extracted latent from a state
        via rssm.get_latent().

        Args:
            latent: Concatenated [h_t; z_t] of shape [B, h_dim + z_dim].
            action: Action tensor of shape [B, action_dim].

        Returns:
            tau_hat: Expected validity horizon [B], in step-space (>= 0).
        """
        logits = self.tau_head(latent, action, stop_grad=False)
        return self.tau_head.decode(logits)

    def interventional(self, rssm_state: State, alt_action: Tensor) -> Tensor:
        """Level 2: do(a = alt_action) — imagine next state under intervention.

        Runs rssm.imagine(alt_action, rssm_state) to obtain the next State,
        then queries tau_head on that state's latent and alt_action.

        The RNG is NOT pinned here — the z_t sample is drawn freely from the
        prior, making this a pure do-operator query: what would τ be if the
        agent took alt_action from the current state?

        Args:
            rssm_state: State(h, z) from which to project forward.
            alt_action: Intervention action [B, action_dim].

        Returns:
            tau_hat: [B], decoded expected horizon from imagined next state.
        """
        # imagine() returns (prior_logits, next_state); we only need next_state
        _, next_state = self.rssm.imagine(alt_action, rssm_state)
        latent = self.rssm.get_latent(next_state)
        logits = self.tau_head(latent, alt_action, stop_grad=False)
        return self.tau_head.decode(logits)

    def counterfactual(
        self,
        rssm_state: State,
        alt_action: Tensor,
        rng_state: dict,
    ) -> Tensor:
        """Level 3: counterfactual — pin z_t to its factual value, change action.

        Restores the RNG to rng_state (captured immediately before the factual
        rssm.imagine() call on the factual trajectory), then runs
        rssm.imagine(alt_action, rssm_state) under the restored RNG. Because
        imagine() calls _sample_z which consumes RNG for the z_t draw, restoring
        to the pre-imagine snapshot guarantees the same z_t noise is drawn —
        effectively abduction of the exogenous noise variable u_z.

        The global RNG is rewound to rng_state before imagine() so that z_t
        noise is identical to the factual trajectory. After the call, the RNG
        is at the position where rng_state was captured (the factual pre-imagine
        snapshot), not at the call-site — callers should be aware of this.

        Args:
            rssm_state: State(h, z) — same as used in the factual trajectory.
            alt_action: Counterfactual action [B, action_dim].
            rng_state: Dict from save_rng_state(), captured before the factual
                       imagine() call so z_t noise is identical to factual.

        Returns:
            tau_hat: [B], decoded expected horizon from counterfactual state.
        """
        # Restore the factual pre-imagine RNG snapshot so _sample_z draws the
        # same z_t noise as in the factual trajectory (abduction of u_z).
        # We leave the RNG at rng_state on exit — the caller's RNG is effectively
        # "rewound" to the factual pre-imagine position, which is the contract
        # the test suite verifies.
        restore_rng_states(rng_state)
        _, next_state = self.rssm.imagine(alt_action, rssm_state)

        latent = self.rssm.get_latent(next_state)
        logits = self.tau_head(latent, alt_action, stop_grad=False)
        return self.tau_head.decode(logits)
