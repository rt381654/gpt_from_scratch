"""
Minimal Training Loop
=====================
Demonstrates how to train the GPT model on a tiny character-level dataset
(Shakespeare).  The entire dataset fits in RAM, so there is no DataLoader —
just random batch sampling from a flat tensor of token ids.

Run:
    python train.py
"""

import torch          # tensor operations and device management
import torch.optim    # optimiser classes (AdamW)

from model import GPT, count_parameters  # our transformer implementation


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# These are intentionally small so the model trains on a laptop CPU/GPU
# within a few minutes.  Scale up for real tasks.

BATCH_SIZE   = 16    # number of independent sequences processed in one step
SEQ_LEN      = 64    # number of tokens per training sequence (context window)
D_MODEL      = 128   # embedding / hidden dimension (width)
N_HEADS      = 4     # number of attention heads (must divide D_MODEL evenly)
N_LAYERS     = 4     # number of stacked transformer blocks (depth)
DROPOUT      = 0.1   # fraction of activations randomly zeroed during training
LR           = 3e-4  # learning rate for AdamW (3e-4 is a classic safe default)
MAX_ITERS    = 2000  # total number of gradient update steps
EVAL_EVERY   = 200   # print loss every this many steps
MAX_NEW_TOKENS = 200 # tokens to generate in the sample at the end

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
# Use a GPU if one is available; otherwise fall back to CPU.
# .to(device) later moves tensors and models to the selected device.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Data — Tiny Shakespeare (character-level)
# ---------------------------------------------------------------------------

def get_data(path: str = "input.txt"):
    """
    Load a plain-text file, build a character-level vocabulary, and encode
    the entire text as a 1-D tensor of integer token ids.

    Character-level tokenisation is the simplest possible scheme:
      - vocab_size == number of unique characters in the file
      - each token is one character
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        # If the file is absent, generate a tiny synthetic dataset so the
        # script still runs end-to-end without downloading anything.
        print("input.txt not found — using a tiny synthetic dataset.")
        text = ("To be, or not to be, that is the question.\n" * 200)

    # --- Build vocabulary ---
    chars = sorted(set(text))          # unique characters in sorted order
    vocab_size = len(chars)            # number of distinct tokens

    # Two lookup dictionaries for encoding (char→int) and decoding (int→char).
    stoi = {ch: i for i, ch in enumerate(chars)}  # string-to-index
    itos = {i: ch for i, ch in enumerate(chars)}  # index-to-string

    # Encode: convert entire text to a list of integer ids, then wrap in a
    # long (int64) tensor which nn.Embedding expects.
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    return data, vocab_size, stoi, itos


# ---------------------------------------------------------------------------
# Batch Sampler
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load data ---
    data, vocab_size, stoi, itos = get_data()
    print(f"Dataset size: {len(data):,} tokens | Vocab size: {vocab_size}")

    # Split into 90% train / 10% validation.
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]

    # --- Build model ---
    model = GPT(
        vocab_size  = vocab_size,
        d_model     = D_MODEL,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        max_seq_len = SEQ_LEN,
        dropout     = DROPOUT,
    ).to(device)  # move all parameters and buffers to the selected device

    print(f"Model parameters: {count_parameters(model):,}")

    # --- Optimiser ---
    # AdamW = Adam with decoupled weight decay.
    # Adam maintains per-parameter adaptive learning rates (good for sparse gradients).
    # Weight decay (L2 regularisation) penalises large weights to prevent overfitting.
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-1)

    # --- Training loop ---
    model.train()  # activates dropout (nn.Dropout is a no-op during eval)

    for step in range(1, MAX_ITERS + 1):
        # Sample a random mini-batch from the training split.
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, device)

        # Forward pass: compute predictions and loss.
        # logits: (B, T, vocab_size), loss: scalar cross-entropy
        logits, loss = model(x, y)

        # Backward pass: compute gradients of loss w.r.t. all parameters.
        optimiser.zero_grad()  # clear gradients from the previous step (they accumulate by default)
        loss.backward()        # backpropagation through the entire graph

        # Gradient clipping: if the global gradient norm exceeds 1.0, scale all
        # gradients down proportionally.  Prevents exploding gradients which can
        # occur in deep networks and derail training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()  # update parameters: θ ← θ - lr * grad

        # --- Periodic evaluation ---
        if step % EVAL_EVERY == 0 or step == 1:
            # Estimate validation loss without updating parameters.
            model.eval()  # deactivates dropout for consistent evaluation
            with torch.no_grad():  # disable gradient tracking to save memory and compute
                val_x, val_y = get_batch(val_data, BATCH_SIZE, SEQ_LEN, device)
                _, val_loss = model(val_x, val_y)
            model.train()  # switch back to training mode

            print(f"Step {step:>5} | train loss: {loss.item():.4f} | val loss: {val_loss.item():.4f}")

    # --- Generate a sample ---
    print("\n--- Generated sample ---")
    model.eval()
    # Start generation from a single newline character as the prompt.
    # unsqueeze(0) adds the batch dimension: (T,) → (1, T).
    start_char = "\n"
    prompt = torch.tensor(
        [stoi.get(start_char, 0)],  # encode the prompt character
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)  # (1, 1)

    # Generate MAX_NEW_TOKENS new tokens autoregressively.
    generated = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8, top_k=40)

    # Decode the integer token ids back to a string.
    # generated[0] selects the first (only) sequence in the batch.
    # .tolist() converts the tensor to a Python list for iteration.
    print("".join(itos[i] for i in generated[0].tolist()))


if __name__ == "__main__":
    # Standard Python guard: only run main() when this script is executed
    # directly (not when it is imported as a module by another script).
    main()
