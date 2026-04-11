"""
Core mathematical utilities for BVH-RSSM.

All functions are pure (no side effects, no state). Shapes follow PyTorch
convention: leading dims are batch dims, trailing dim is feature/class dim.
"""
import torch
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    """Symmetric log: sign(x) * log(|x| + 1).

    Compresses large values while preserving small values near origin.
    Used for: observation inputs to encoder, decoder targets, reward targets.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of same shape as x, with values in symlog space.

    Complexity: O(N) where N = x.numel(). No allocations beyond output.
    Side effects: None.
    """
    return torch.sign(x) * torch.log(x.abs() + 1.0)


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1).

    Applied at output time to recover original scale from symlog-encoded values.
    Exact inverse: symexp(symlog(x)) == x for all finite x.

    Args:
        x: Input tensor in symlog space, arbitrary shape.

    Returns:
        Tensor of same shape as x, in original scale.

    Complexity: O(N). No allocations beyond output.
    Side effects: None.
    """
    return torch.sign(x) * (x.abs().exp() - 1.0)


def symlog_bins(n_bins: int = 255, lo: float = -20.0, hi: float = 20.0) -> Tensor:
    """Evenly-spaced bin centers in symlog space.

    Args:
        n_bins: Number of bins (DreamerV3 default: 255).
        lo: Lower bound in symlog space.
        hi: Upper bound in symlog space.

    Returns:
        Tensor of shape [n_bins] sorted ascending, on CPU.

    Complexity: O(n_bins). Allocates a single 1-D tensor.
    Side effects: None.
    """
    return torch.linspace(lo, hi, n_bins)


def twohot(x: Tensor, bins: Tensor) -> Tensor:
    """Encode scalar(s) as soft two-hot vectors over a bin grid.

    Two adjacent bins receive fractional weights summing to 1. All other
    bins receive 0. This decouples gradient magnitude from prediction scale
    and enables distributional regression without discretization artifacts.

    Values outside [bins[0], bins[-1]] are clamped to the nearest valid
    bin pair (lower_idx in [0, B-2]) so the encoding remains valid.

    Args:
        x: Scalar targets of shape [*batch]. Should be in symlog space.
        bins: Sorted bin centers of shape [B].

    Returns:
        Two-hot encoded tensor of shape [*batch, B].
        Each row sums to 1.0, has exactly 2 non-zero entries (or 1 if x
        lands exactly on a bin boundary at the edge).

    Complexity: O(N * log B) for searchsorted, O(N * B) memory.
    Side effects: None.
    """
    batch_shape = x.shape
    B = bins.shape[0]
    x_flat = x.reshape(-1)  # [N]

    # Find lower bin index for each value (clamp to valid range).
    # searchsorted with right=True returns insertion point such that
    # bins[idx-1] <= x_flat < bins[idx], so lower_idx = idx - 1.
    lower_idx = torch.searchsorted(bins.contiguous(), x_flat.contiguous(), right=True) - 1
    lower_idx = lower_idx.clamp(0, B - 2)
    upper_idx = lower_idx + 1

    lower_val = bins[lower_idx]   # [N]
    upper_val = bins[upper_idx]   # [N]

    # Linear interpolation: weight_upper = (x - lower) / (upper - lower).
    # Clamped to [0,1] to handle x outside bin range (already clamped by
    # lower_idx clamp, but guard against fp edge cases).
    span = (upper_val - lower_val).clamp(min=1e-8)
    weight_upper = (x_flat - lower_val) / span          # [N]
    weight_upper = weight_upper.clamp(0.0, 1.0)
    weight_lower = 1.0 - weight_upper                   # [N]

    out = torch.zeros(x_flat.shape[0], B, device=x.device, dtype=x.dtype)
    out.scatter_(1, lower_idx.unsqueeze(1), weight_lower.unsqueeze(1))
    out.scatter_(1, upper_idx.unsqueeze(1), weight_upper.unsqueeze(1))

    return out.reshape(*batch_shape, B)


def twohot_decode(probs: Tensor, bins: Tensor) -> Tensor:
    """Decode a probability distribution over bins to its expected value.

    Computes E[X] = sum_i p_i * bins_i under the categorical distribution.
    Input must already be a valid probability distribution (non-negative,
    sums to 1). If you have raw logits, apply softmax before calling this.

    Design rationale: keeping softmax external avoids the double-softmax
    pitfall where valid probs passed through softmax again produce wrong
    results. The caller controls normalisation.

    Args:
        probs: Probability tensor of shape [*batch, B]. Must sum to 1
               along the last dimension. No internal normalisation applied.
        bins: Sorted bin centers of shape [B].

    Returns:
        Expected value tensor of shape [*batch].

    Complexity: O(N * B). Side effects: None.
    """
    return (probs * bins.to(probs.device)).sum(-1)


def unimix(logits: Tensor, eps: float = 0.01) -> Tensor:
    """Mix categorical distribution with uniform: (1-eps)*q_neural + eps*q_uniform.

    Ensures minimal probability mass everywhere — prevents log(0) in KL
    divergence computation and ensures stable gradients through the entire
    categorical latent space.

    Args:
        logits: Unnormalized logits of shape [*batch, n_classes].
        eps: Mixing coefficient (DreamerV3 default: 0.01).
             Each class gets at least eps/n_classes probability mass.

    Returns:
        Mixed probability tensor of shape [*batch, n_classes]. Sums to 1.

    Complexity: O(N). Side effects: None.
    """
    probs = logits.softmax(-1)
    n_classes = logits.shape[-1]
    uniform = torch.ones_like(probs) / n_classes
    return (1.0 - eps) * probs + eps * uniform
