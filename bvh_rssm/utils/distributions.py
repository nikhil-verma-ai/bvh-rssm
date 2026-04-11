"""
Categorical distribution utilities for BVH-RSSM.

Implements straight-through gradient estimator for discrete sampling,
which is required for backpropagation through categorical latent states z_t.

The STE is implemented via a custom autograd.Function that routes the
identity gradient directly through logits. This guarantees gradient flow
regardless of the downstream loss shape, unlike the naive
`one_hot + probs - probs.detach()` formulation whose gradient vanishes
whenever the loss is invariant under permutation of softmax outputs
(e.g. sum loss, since sum(softmax) = 1 is constant w.r.t. logits).

Identity STE through logits: backward passes grad_output unchanged back
to logits. In real training the loss depends on which class was selected
and the corresponding entry gradient flows to that logit correctly.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


class _STEFunction(torch.autograd.Function):
    """Custom autograd function implementing the straight-through estimator.

    Forward: returns the hard one-hot sample (no gradient graph).
    Backward: identity pass — routes grad_output to logits unchanged.

    This avoids the vanishing gradient problem of the naive formulation
    one_hot + probs - probs.detach(), which produces zero gradient when
    the loss is invariant to the softmax sum (constant = 1).
    """

    @staticmethod
    def forward(ctx, one_hot: Tensor, logits: Tensor) -> Tensor:  # type: ignore[override]
        # one_hot is the hard discrete sample; logits carries the grad graph.
        return one_hot

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        # Identity STE: pass gradient to logits as-is; one_hot needs no grad.
        return None, grad_output


def straight_through_sample(logits: Tensor) -> Tensor:
    """Sample one-hot from categorical with straight-through gradient.

    Forward pass: argmax (deterministic one-hot).
    Backward pass: identity gradient flows back to logits unchanged.
    This is the STE used for discrete latent states z_t in DreamerV3.

    Args:
        logits: Unnormalized logits of shape [*batch, n_cats, n_classes].
                For z_t: shape [B, 32, 32].

    Returns:
        One-hot tensor of same shape. Forward: hard argmax one-hot.
        Backward: identity STE through logits.
    """
    probs = logits.softmax(-1)
    indices = probs.argmax(-1)                                          # [*batch, n_cats]
    one_hot = F.one_hot(indices, num_classes=logits.shape[-1]).float()  # [*batch, n_cats, n_classes]
    return _STEFunction.apply(one_hot, logits)


def sample_categorical(logits: Tensor) -> Tensor:
    """Sample from categorical distribution with straight-through gradient (training mode).

    Uses multinomial sampling (stochastic) rather than argmax. Produces
    different samples each call. Gradient flows via identity STE through logits.

    Args:
        logits: Unnormalized logits of shape [*batch, n_classes].

    Returns:
        One-hot tensor of same shape with straight-through gradient through logits.
    """
    probs = logits.softmax(-1)
    batch_shape = probs.shape[:-1]
    n_classes = probs.shape[-1]

    # Flatten for multinomial, then restore batch shape
    flat_probs = probs.reshape(-1, n_classes)
    indices = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # [N]
    indices = indices.reshape(batch_shape)                               # [*batch]

    one_hot = F.one_hot(indices, num_classes=n_classes).float()         # [*batch, n_classes]
    return _STEFunction.apply(one_hot, logits)
