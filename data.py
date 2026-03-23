"""
data.py — Toy Dataset and Batch Sampler
=========================================
Responsible for two things:
  1. Building the toy arithmetic corpus that the model trains on.
  2. Sampling random (input, target) mini-batches from a token tensor.

Keeping data logic here means train.py stays focused on the training loop
and model.py stays focused on the architecture.
"""

import torch                 # for torch.tensor and torch.randint / torch.stack
from tokenizer import Tokenizer  # character-level vocab + encode/decode


def build_toy_dataset():
    """
    Generate all single-digit addition facts (a+b=c for a,b in 0-9),
    repeated to form a corpus of ~56 000 characters, and return a tokenized
    tensor plus the fitted Tokenizer.

    Returns:
        data      : 1-D LongTensor of token ids for the full corpus
        tokenizer : fitted Tokenizer (holds vocab_size, encode, decode)
    """
    # Build every addition fact as a short string: "a+b=c, "
    # Using all pairs (a, b) where a, b ∈ {0…9} gives 100 unique facts.
    facts = []
    for a in range(10):       # first operand 0-9
        for b in range(10):   # second operand 0-9
            c = a + b         # correct sum (may be two digits, e.g. 9+9=18)
            facts.append(f"{a}+{b}={c}, ")

    # Repeat many times so the corpus is large enough for stable mini-batch training.
    # ~700 characters per pass × 80 repeats ≈ 56 000 characters total.
    text = "".join(facts * 80)

    # Fit the tokenizer on the corpus text — builds the character vocabulary.
    tokenizer = Tokenizer(text)

    # Encode: convert entire text to a list of integer ids, then wrap in a
    # long (int64) tensor which nn.Embedding expects.
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    return data, tokenizer


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str):
    """
    Sample a random mini-batch of (input, target) sequence pairs.

    Language modelling is a one-step-ahead prediction task:
      - inputs[b]  = data[i   : i+seq_len]
      - targets[b] = data[i+1 : i+seq_len+1]
    Each target token is the token that follows the corresponding input token.
    The model must learn to predict target[t] given inputs[0..t].

    Returns:
        x: (batch_size, seq_len) input token ids
        y: (batch_size, seq_len) target token ids (shifted by one position)
    """
    # Sample batch_size random starting positions, leaving room for seq_len+1 tokens.
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))

    # Stack individual sequences into (batch_size, seq_len) tensors.
    x = torch.stack([data[i    : i + seq_len    ] for i in ix])
    y = torch.stack([data[i + 1: i + seq_len + 1] for i in ix])

    # Move to the compute device (GPU/CPU).
    return x.to(device), y.to(device)
