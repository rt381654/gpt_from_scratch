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
"""

import torch                              # tensor operations and autograd
import torch.nn as nn                     # building-block layers
import torch.nn.functional as F           # cross_entropy, softmax

from positional_embedding import PositionalEmbedding  # token + position embeddings
from block import TransformerBlock                     # single transformer layer


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
        loss = None
        if targets is not None:
            # Cross-entropy expects (N, C) logits and (N,) targets.
            # Flatten both: (B*T, vocab_size) and (B*T,).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),                  # (B*T,)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,  # (B, T) prompt token ids
        max_new_tokens: int,           # number of tokens to generate
        temperature:    float = 1.0,   # >1 = more random, <1 = more peaked/greedy
        top_k:          int   = None,  # if set, restrict sampling to top-k logits
    ) -> torch.Tensor:
        """
        Autoregressive generation: repeatedly predict the next token and
        append it to the sequence.

        Args:
            idx:            Prompt token ids, shape (B, T).
            max_new_tokens: How many new tokens to sample.
            temperature:    Scales logits before softmax (controls randomness).
            top_k:          If set, zero out all but the top-k logit positions.
        Returns:
            idx: Extended token id tensor, shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Truncate the context to the last max_seq_len tokens if it has grown
            # beyond what the positional embedding table supports.
            max_seq_len = self.embedding.pos_emb.num_embeddings
            idx_cond = idx[:, -max_seq_len:]  # (B, T') where T' <= max_seq_len

            # Forward pass — we only care about the logits of the last position
            # because that is the next-token prediction.
            logits, _ = self(idx_cond)          # (B, T', vocab_size)
            logits = logits[:, -1, :]           # (B, vocab_size) — last position only

            # Apply temperature: dividing by a small number sharpens the distribution
            # (makes the model more confident); dividing by >1 flattens it (more random).
            logits = logits / temperature

            # Top-k filtering: set all logits below the k-th largest to -inf so they
            # get zero probability after softmax.  Encourages coherent generations.
            if top_k is not None:
                # torch.topk returns the k largest values; v[:, [-1]] is the k-th largest.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Convert logits to probabilities.
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample one token index from the probability distribution.
            # multinomial draws without replacement; num_samples=1 gives one token per batch.
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new token to the running sequence and continue.
            idx = torch.cat([idx, next_token], dim=1)  # (B, T+1)

        return idx  # (B, T + max_new_tokens)


# ---------------------------------------------------------------------------
# Convenience: count trainable parameters
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    # p.numel() gives the total number of elements in a parameter tensor.
    # We only count parameters where requires_grad=True (learnable ones).
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
