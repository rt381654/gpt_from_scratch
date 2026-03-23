"""
positional_embedding.py — Token + Positional Embeddings
=========================================================
Converts a batch of integer token ids into continuous vector representations
by combining two learned embedding tables:

  1. Token embedding   — maps each token id to a d_model vector.
                         Tokens with similar meaning end up close in this space.
  2. Positional embedding — maps each sequence position (0, 1, 2, …) to a
                         d_model vector.  This injects order information because
                         self-attention is otherwise position-agnostic (a set op).

The two embeddings are summed element-wise, then dropout is applied.
"""

import torch                  # for torch.arange
import torch.nn as nn         # Embedding, Dropout, Module


class PositionalEmbedding(nn.Module):
    """
    Learnable token embedding + learnable positional embedding, summed and
    passed through dropout.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float):
        """
        Args:
            vocab_size:  Number of unique tokens (size of the token vocabulary).
            d_model:     Embedding dimension (width of the model).
            max_seq_len: Maximum number of positions that can be embedded.
            dropout:     Dropout probability applied after combining embeddings.
        """
        super().__init__()

        # Learnable lookup table: maps each integer token id to a dense vector
        # of size d_model.  The model learns which tokens are semantically similar
        # by making their vectors close in this space.
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Pure transformers have no notion of order — attention is a set operation.
        # A learnable positional embedding adds position-specific signals so the
        # model knows *where* in the sequence each token appears.
        # Shape: (max_seq_len, d_model); position p gets its own vector.
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Dropout applied after combining token + positional embeddings.
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: Integer token id tensor, shape (B, T).
        Returns:
            Embedding tensor of shape (B, T, d_model).
        """
        # Look up token embeddings: (B, T) → (B, T, d_model)
        tok = self.token_emb(idx)

        # Create a position index tensor [0, 1, …, T-1] and look up positional embeddings.
        # arange produces a 1-D tensor; unsqueeze(0) adds a batch dim → (1, T).
        # Embedding lookup: (1, T) → (1, T, d_model); broadcasts over the batch.
        T   = idx.size(1)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        pos = self.pos_emb(pos)                                 # (1, T, d_model)

        # Add token and positional embeddings element-wise; apply dropout.
        return self.dropout(tok + pos)  # (B, T, d_model)
