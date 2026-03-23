"""
generate.py — Autoregressive Token Sampling
=============================================
Generates new tokens from a trained GPT model one step at a time.

The process is autoregressive: at each step the model sees all tokens
generated so far (the "context") and predicts a probability distribution
over the next token; one token is sampled from that distribution and
appended to the context before the next step.

Two sampling controls are provided:
  temperature — scales logits before softmax.
                < 1.0 sharpens the distribution (more greedy / confident).
                > 1.0 flattens it (more random / creative).
  top_k       — restricts sampling to the k most likely tokens at each step,
                preventing the model from picking very unlikely tokens.
"""

import torch                          # tensor operations
import torch.nn.functional as F       # F.softmax


@torch.no_grad()  # disable gradient tracking — we are doing inference, not training
def generate(
    model,                        # trained GPT instance
    idx:            torch.Tensor, # (B, T) prompt token ids
    max_new_tokens: int,          # number of tokens to generate
    temperature:    float = 1.0,  # >1 = more random, <1 = more peaked/greedy
    top_k:          int   = None, # if set, restrict sampling to top-k logits
) -> torch.Tensor:
    """
    Autoregressively sample `max_new_tokens` tokens and append them to `idx`.

    Args:
        model:          Trained GPT model (must be in eval mode for deterministic output).
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
        max_seq_len = model.embedding.pos_emb.num_embeddings
        idx_cond = idx[:, -max_seq_len:]  # (B, T') where T' <= max_seq_len

        # Forward pass — we only care about the logits of the last position
        # because that is the next-token prediction.
        logits, _ = model(idx_cond)   # (B, T', vocab_size)
        logits = logits[:, -1, :]     # (B, vocab_size) — last position only

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
