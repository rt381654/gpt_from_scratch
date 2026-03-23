"""
attention.py — Causal Multi-Head Self-Attention
================================================
Implements the core attention mechanism used in decoder-only transformers.

Key ideas:
  - Each token computes a query (what am I looking for?), a key (what do I
    offer?), and a value (what information do I carry?).
  - Dot products between queries and keys score how relevant each past token
    is to the current token.
  - A causal mask ensures token i can only attend to positions 0 … i, never
    to future positions — this is what makes generation autoregressive.
  - "Multi-head" runs several independent attention operations in parallel,
    each in a lower-dimensional subspace, then concatenates the results.
"""

import math                      # math.sqrt for the attention scaling factor
import torch                     # tensor operations
import torch.nn as nn            # Module base class, Linear, Dropout
import torch.nn.functional as F  # F.softmax


class CausalSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention with a causal mask.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        """
        Args:
            d_model:     Total embedding dimension (must be divisible by n_heads).
            n_heads:     Number of parallel attention heads.
            max_seq_len: Maximum sequence length the model will ever see.
            dropout:     Dropout probability applied after the attention weights.
        """
        super().__init__()

        # Every head works in a subspace of size head_dim = d_model / n_heads.
        # Splitting the full d_model dimension across heads keeps total compute
        # the same as a single-head attention with d_model dimensions.
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads  # dimension per head
        self.d_model  = d_model

        # Three separate linear projections produce Queries, Keys, and Values
        # from the input embeddings.  All three map d_model → d_model so that
        # after splitting into n_heads each head gets head_dim features.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # query projection
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # key   projection
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # value projection

        # Output projection: recombines the concatenated head outputs back into
        # a single d_model-dimensional vector.
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout applied to attention weights — regularises which keys each
        # query attends to, preventing the model from over-relying on specific positions.
        self.attn_dropout = nn.Dropout(dropout)

        # Pre-compute the causal mask once and register it as a buffer so it
        # moves to the correct device automatically with .to(device) / .cuda().
        # torch.tril creates a lower-triangular matrix:
        #   mask[i, j] = 1  if j <= i  (token j is visible to token i)
        #   mask[i, j] = 0  if j >  i  (future token — blocked)
        # Shape: (1, 1, max_seq_len, max_seq_len) — the two leading 1-dims let
        # PyTorch broadcast over the batch and head dimensions.
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        Returns:
            Output tensor of the same shape (batch, seq_len, d_model).
        """
        B, T, C = x.shape  # batch size, sequence length, embedding dim (C == d_model)

        # --- Project inputs into Q, K, V ---
        # Each projection: (B, T, d_model) → (B, T, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # --- Split into multiple heads ---
        # Reshape from (B, T, d_model) → (B, T, n_heads, head_dim)
        # then transpose to (B, n_heads, T, head_dim) so that the batch
        # and head dimensions are leading — allows batched matmul over heads.
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # --- Scaled dot-product attention ---
        # Compute raw attention scores: how much does each query match each key?
        # Q: (B, n_heads, T, head_dim)  @  K^T: (B, n_heads, head_dim, T)
        # Result scores: (B, n_heads, T, T)
        scores = Q @ K.transpose(-2, -1)

        # Scale by 1/sqrt(head_dim) to prevent dot products from growing too
        # large (which would push softmax into regions with near-zero gradients).
        scores = scores / math.sqrt(self.head_dim)

        # Apply causal mask: positions where mask == 0 (future tokens) are set
        # to -inf so that after softmax they contribute exactly 0 to the output.
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax turns raw scores into a probability distribution over the T
        # key positions for each query position.
        attn_weights = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)

        # Randomly zero out some attention weights during training to regularise.
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: each output position is a mix of all value
        # vectors, weighted by how much the query attended to each key.
        # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) → (B, n_heads, T, head_dim)
        out = attn_weights @ V

        # --- Recombine heads ---
        # Transpose back: (B, n_heads, T, head_dim) → (B, T, n_heads, head_dim)
        # contiguous() is required before view() to ensure memory layout is contiguous.
        # view merges the last two dims: (B, T, n_heads * head_dim) == (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Final linear projection mixes information across heads.
        return self.out_proj(out)  # (B, T, d_model)
