"""
bvh_rssm.causal — Pearl causal hierarchy and adaptive policy routing.

Public API:
  CausalAttributor  — Level 1/2/3 causal attribution over RSSM latents
  AdaptivePolicyRouter — Survival-curve-based validity state classification
  RouterState        — HIGH / DIM / STALE enum
"""
from bvh_rssm.causal.attribution import CausalAttributor
from bvh_rssm.causal.router import AdaptivePolicyRouter, RouterState

__all__ = ["CausalAttributor", "AdaptivePolicyRouter", "RouterState"]
