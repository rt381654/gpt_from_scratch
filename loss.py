"""
loss.py — Loss Computation
===========================
Computes the cross-entropy language modelling loss from logits and targets.

Keeping this here separates the "how do we score the model?" question from
the "what does the model compute?" question in model.py.
"""

import torch
import torch.nn.functional as F  # F.cross_entropy


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss between predicted logits and target token ids.

    Language modelling loss: at every position t, the model predicts a
    distribution over the vocabulary; the loss penalises how much probability
    mass is assigned away from the correct next token.

    Args:
        logits:  Raw (unnormalised) scores, shape (B, T, vocab_size).
        targets: Ground-truth next-token ids, shape (B, T).
    Returns:
        Scalar cross-entropy loss averaged over all (B * T) positions.
    """
    # F.cross_entropy expects 2-D logits (N, C) and 1-D targets (N,).
    # We flatten the batch and time dimensions together: B*T positions,
    # each with vocab_size scores and one correct target id.
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
        targets.view(-1),                  # (B*T,)
    )
