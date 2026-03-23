"""
block.py — Transformer Block (single layer)
=============================================
One transformer layer = Attention sublayer + FFN sublayer, each wrapped
with Pre-LayerNorm and a residual (skip) connection.

Design choices explained:
  Pre-LayerNorm (norm applied *before* each sublayer, not after):
    - The original "Attention Is All You Need" paper used post-norm.
    - GPT-2 and most modern LLMs switched to pre-norm because it gives
      more stable gradients during early training.

  Residual connections  output = x + sublayer(norm(x)):
    - Allow gradients to flow directly from the loss back to early layers
      without passing through every sublayer multiplication.
    - Let the network learn incremental refinements rather than full
      transformations at each layer — easier to optimise.
"""

import torch.nn as nn         # LayerNorm, Dropout, Module
import torch                  # Tensor type hint

from attention import CausalSelfAttention  # multi-head causal self-attention
from mlp import FeedForward                # position-wise feed-forward network


class TransformerBlock(nn.Module):
    """
    A single transformer layer combining causal self-attention and an MLP,
    each preceded by layer normalisation and followed by a residual connection.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        """
        Args:
            d_model:     Embedding / hidden dimension.
            n_heads:     Number of attention heads.
            max_seq_len: Maximum sequence length (passed to attention for mask).
            dropout:     Dropout probability used in attention and FFN.
        """
        super().__init__()

        # LayerNorm normalises each token's feature vector to zero mean / unit variance.
        # Two separate norms — one before attention, one before FFN.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # The two sublayers that make up a transformer block.
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ffn  = FeedForward(d_model, dropout)

        # Dropout on each sublayer's output before adding to the residual.
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, d_model).
        Returns:
            Output tensor of the same shape (B, T, d_model).
        """
        # Pre-norm attention with residual connection.
        # norm(x) stabilises the input; drop(...) regularises the output;
        # adding x is the skip connection that preserves the original signal.
        x = x + self.drop(self.attn(self.norm1(x)))

        # Pre-norm FFN with residual connection.
        x = x + self.drop(self.ffn(self.norm2(x)))

        return x  # (B, T, d_model)
