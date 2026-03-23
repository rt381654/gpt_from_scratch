"""
mlp.py — Position-wise Feed-Forward Network
=============================================
The MLP sublayer applied independently at every sequence position after
the attention sublayer inside each transformer block.

Why a separate MLP?
  - Attention aggregates information *across* positions (which tokens to mix).
  - The MLP transforms information *within* each position (what to do with it).
  - Together they cover both the relational and the representational work.

The design (expand → activate → contract) follows the original Transformer
paper and GPT-2: d_model → 4*d_model → d_model with GELU activation.
"""

import torch.nn as nn  # Sequential, Linear, GELU, Dropout


class FeedForward(nn.Module):
    """
    Two-layer fully-connected network applied independently to every position.

    The intermediate dimension is 4× the model dimension — this large
    expansion gives the FFN capacity to store factual knowledge and perform
    more complex per-token transformations than attention alone.
    """

    def __init__(self, d_model: int, dropout: float):
        """
        Args:
            d_model: Input and output dimension.
            dropout: Dropout probability applied after the activation.
        """
        super().__init__()

        # Expand from d_model → 4*d_model, apply GELU, then project back.
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # expand: (B, T, d_model) → (B, T, 4*d_model)
            nn.GELU(),                         # smooth non-linearity; better gradient flow than ReLU
            nn.Dropout(dropout),               # randomly zero activations to reduce overfitting
            nn.Linear(4 * d_model, d_model),   # contract back: (B, T, 4*d_model) → (B, T, d_model)
        )

    def forward(self, x):
        return self.net(x)  # applied position-wise; shape is unchanged (B, T, d_model)
