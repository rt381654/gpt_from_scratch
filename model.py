"""
Barebone GPT-style Decoder-Only Transformer
============================================
A from-scratch implementation of the architecture introduced in
"Attention Is All You Need" (Vaswani et al., 2017) and refined in
the GPT series (Radford et al., 2018-2020).

A decoder-only transformer differs from the full encoder-decoder design:
  - There is no separate encoder stack.
  - Attention is always CAUSAL (masked): each token can only attend to
    itself and tokens that came before it. This is what makes it suitable
    for autoregressive language modelling — predicting the next token.

Model internals are split into focused modules:
  attention.py            — CausalSelfAttention
  mlp.py                  — FeedForward
  positional_embedding.py — PositionalEmbedding
  block.py                — TransformerBlock (composes attention + mlp)
  loss.py                 — compute_loss (cross-entropy over logits)
  generate.py             — generate (autoregressive token sampling)
"""

import torch                              # tensor operations and autograd
import torch.nn as nn                     # building-block layers

from positional_embedding import PositionalEmbedding  # token + position embeddings
from block import TransformerBlock                     # single transformer layer
from loss import compute_loss                          # cross-entropy loss computation


class GPT(nn.Module):
    """
    Decoder-only transformer language model (GPT-style).

    Forward pass:
      token ids → token embeddings + positional embeddings
                → N transformer blocks
                → final layer norm
                → linear projection to vocabulary logits

    During inference the model is used autoregressively: it generates one
    token at a time, feeding each predicted token back as input for the next
    step.
    """

    def __init__(
        self,
        vocab_size:  int,    # number of unique tokens (size of the vocabulary)
        d_model:     int,    # embedding dimension (width of the model)
        n_heads:     int,    # number of attention heads per layer
        n_layers:    int,    # number of stacked transformer blocks (depth)
        max_seq_len: int,    # maximum context window length
        dropout:     float,  # dropout probability used throughout
    ):
        super().__init__()

        # --- Token + Positional Embeddings ---
        # Combines the token embedding table and the positional embedding table
        # into a single module (see positional_embedding.py).
        self.embedding = PositionalEmbedding(vocab_size, d_model, max_seq_len, dropout)

        # --- Stack of Transformer Blocks ---
        # nn.ModuleList registers all blocks as proper submodules so their
        # parameters appear in model.parameters() and are moved by .to(device).
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)   # one block per layer; each has its own weights
        ])

        # --- Final Layer Norm ---
        # Applied after all transformer blocks, before the output projection.
        # Stabilises the magnitude of the representations reaching the head.
        self.norm = nn.LayerNorm(d_model)

        # --- Language Model Head ---
        # Projects the final d_model representations to raw scores (logits)
        # over the entire vocabulary.  No bias is typical for large LMs.
        # Shape: d_model → vocab_size  (applied independently at each position)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share the token embedding matrix with the lm_head.
        # Tokens that are similar in embedding space will also have similar
        # output logits.  Reduces parameters and often improves perplexity.
        self.lm_head.weight = self.embedding.token_emb.weight

        # --- Weight Initialisation ---
        # Apply custom initialisation to all submodules.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        GPT-2-style initialisation:
          - Linear and Embedding weights: N(0, 0.02)
          - Biases: zero
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx:     torch.Tensor,           # (B, T)  integer token ids
        targets: torch.Tensor = None,    # (B, T)  target ids for computing loss (optional)
    ):
        """
        Args:
            idx:     Token index tensor, shape (B, T).
            targets: Shifted token index tensor for teacher-forcing loss, shape (B, T).
                     If None, only logits are returned (inference mode).
        Returns:
            logits: Raw scores over the vocabulary, shape (B, T, vocab_size).
            loss:   Cross-entropy loss scalar, or None if targets not provided.
        """
        # --- Embeddings ---
        # Combines token lookup + positional lookup + dropout (see positional_embedding.py).
        x = self.embedding(idx)  # (B, T, d_model)

        # --- Transformer Blocks ---
        # Pass through each block sequentially; x is refined at every layer.
        for block in self.blocks:
            x = block(x)  # (B, T, d_model)

        # --- Final Norm + LM Head ---
        x = self.norm(x)                  # normalise final representations
        logits = self.lm_head(x)          # project to vocab: (B, T, vocab_size)

        # --- Loss (optional) ---
        # Delegated to loss.py — see compute_loss() for details.
        loss = compute_loss(logits, targets) if targets is not None else None

        return logits, loss


# ---------------------------------------------------------------------------
# Convenience: count trainable parameters
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    # p.numel() gives the total number of elements in a parameter tensor.
    # We only count parameters where requires_grad=True (learnable ones).
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
