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
"""

import math          # math.sqrt used in scaled dot-product attention
import torch         # core tensor operations and autograd engine
import torch.nn as nn  # building-block neural network layers (Linear, Dropout, …)
import torch.nn.functional as F  # stateless functions: softmax, gelu, …


# ---------------------------------------------------------------------------
# 1.  Multi-Head Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Scaled dot-product self-attention with a causal (look-ahead) mask.

    "Multi-head" means we run several attention heads in parallel, each
    learning to focus on different positional / semantic relationships,
    then concatenate their outputs.

    Causal mask: token at position i is only allowed to attend to positions
    0 … i.  Positions i+1, i+2, … are masked out (set to -inf before softmax
    so they become 0 after softmax).  This prevents the model from "cheating"
    by looking at future tokens during training.
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
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # dimension per head
        self.d_model = d_model

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


# ---------------------------------------------------------------------------
# 2.  Position-wise Feed-Forward Network (FFN)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Two-layer fully-connected network applied independently to every position.

    The intermediate dimension is typically 4× the model dimension — this
    large expansion gives the FFN capacity to store factual knowledge and
    perform more complex per-token transformations than attention alone.

    GPT-2 uses GELU activation (smoother alternative to ReLU).
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # applied position-wise; shape is unchanged (B, T, d_model)


# ---------------------------------------------------------------------------
# 3.  Transformer Block (one layer)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single transformer layer = Self-Attention + FFN, each wrapped with
    Pre-LayerNorm and a residual (skip) connection.

    Pre-LayerNorm (norm before sublayer) stabilises training compared to
    the original post-norm described in Vaswani et al.

    Residual connections: output = x + sublayer(norm(x))
      - Let gradients flow directly from loss back to early layers.
      - Allow the network to learn incremental refinements rather than
        complete transformations at each layer.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
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
        # Pre-norm attention with residual connection.
        # norm(x) stabilises the input; drop(...) regularises the output;
        # adding x is the skip connection that preserves the original signal.
        x = x + self.drop(self.attn(self.norm1(x)))

        # Pre-norm FFN with residual connection.
        x = x + self.drop(self.ffn(self.norm2(x)))

        return x  # (B, T, d_model)


# ---------------------------------------------------------------------------
# 4.  Full GPT Model
# ---------------------------------------------------------------------------

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

        # --- Token Embedding ---
        # Learnable lookup table: maps each integer token id to a dense vector
        # of size d_model.  The model learns which tokens are semantically similar
        # by making their vectors close in this space.
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # --- Positional Embedding ---
        # Pure transformers have no notion of order — attention is a set operation.
        # A learnable positional embedding adds position-specific signals so the
        # model knows *where* in the sequence each token appears.
        # Shape: (max_seq_len, d_model); position p gets its own vector.
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Dropout applied after combining token + positional embeddings.
        self.emb_drop = nn.Dropout(dropout)

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
        self.lm_head.weight = self.token_emb.weight

        # --- Weight Initialisation ---
        # Apply custom initialisation to all submodules.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        GPT-2-style initialisation:
          - Linear and Embedding weights: N(0, 0.02)
          - Biases: zero
          - Residual projection layers are scaled by 1/sqrt(n_layers) to
            prevent variance from growing as signals pass through many layers.
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
        B, T = idx.shape

        # --- Embeddings ---
        # Look up token embeddings: (B, T) → (B, T, d_model)
        tok = self.token_emb(idx)

        # Create a position index tensor [0, 1, …, T-1] and look up positional embeddings.
        # arange produces a 1-D tensor; unsqueeze(0) adds a batch dim → (1, T).
        # Embedding lookup: (1, T) → (1, T, d_model); broadcasts over the batch.
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos = self.pos_emb(pos)  # (1, T, d_model)

        # Add token and positional embeddings element-wise; apply dropout.
        x = self.emb_drop(tok + pos)  # (B, T, d_model)

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
        idx:         torch.Tensor,  # (B, T) prompt token ids
        max_new_tokens: int,        # number of tokens to generate
        temperature: float = 1.0,   # >1 = more random, <1 = more peaked/greedy
        top_k:       int   = None,  # if set, restrict sampling to top-k logits
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
            max_seq_len = self.pos_emb.num_embeddings
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
# 5.  Convenience: count trainable parameters
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    # p.numel() gives the total number of elements in a parameter tensor.
    # We only count parameters where requires_grad=True (learnable ones).
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
