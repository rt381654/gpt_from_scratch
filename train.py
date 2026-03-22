"""
Training and Evaluation on a Toy Dataset
=========================================
Trains the GPT model on a small built-in toy dataset — no external files needed.

The toy dataset is a repeating arithmetic sequence of the form:
    "1+1=2, 1+2=3, 2+1=3, 1+3=4, ..."
This gives the model a simple, learnable pattern that shows clear loss
improvement within a few hundred steps, making it easy to verify that
training is working correctly.

Run:
    python train.py
"""

import torch         # tensor operations and autograd
import torch.nn.functional as F  # for cross_entropy used in evaluate()

from model import GPT, count_parameters


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE     = 32    # sequences per gradient step
SEQ_LEN        = 32    # context window (tokens per sequence)
D_MODEL        = 64    # embedding / hidden dimension
N_HEADS        = 4     # attention heads (must divide D_MODEL)
N_LAYERS       = 3     # transformer blocks stacked
DROPOUT        = 0.1   # dropout probability
LR             = 1e-3  # learning rate (slightly higher for small model / toy data)
MAX_ITERS      = 500   # total training steps
EVAL_EVERY     = 50    # evaluate and print metrics every N steps
EVAL_ITERS     = 20    # number of batches averaged when estimating eval loss
MAX_NEW_TOKENS = 60    # tokens to generate in the final sample

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Toy Dataset
# ---------------------------------------------------------------------------
# We build a small character-level arithmetic corpus entirely in memory.
# Each example looks like "3+5=8" — simple enough that a tiny model can
# memorise the pattern and show clear loss curves within ~500 steps.

def build_toy_dataset():
    """
    Generate all single-digit addition facts (a + b = c, a and b in 0-9),
    repeated enough times to form a corpus of ~50 k characters.

    Returns:
        data       : 1-D LongTensor of token ids covering the entire corpus
        vocab_size : number of unique characters
        stoi       : char → int encoding dict
        itos       : int  → char decoding dict
    """
    # Build every addition fact as a short string: "a+b=c, "
    # Using all pairs (a, b) where a, b ∈ {0…9} gives 100 unique facts.
    facts = []
    for a in range(10):          # first operand 0-9
        for b in range(10):      # second operand 0-9
            c = a + b            # correct sum (0-18; may be two digits)
            facts.append(f"{a}+{b}={c}, ")  # e.g. "3+7=10, "

    # Repeat the fact list many times to give the model plenty of examples.
    # ~600 characters per pass × 80 repeats ≈ 48 000 characters total.
    corpus = "".join(facts * 80)

    # --- Character-level vocabulary ---
    chars      = sorted(set(corpus))          # unique chars, sorted for reproducibility
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}  # char → int
    itos = {i: ch for i, ch in enumerate(chars)}  # int  → char

    # Encode the entire corpus as a LongTensor of token ids.
    data = torch.tensor([stoi[ch] for ch in corpus], dtype=torch.long)

    return data, vocab_size, stoi, itos


# ---------------------------------------------------------------------------
# Batch Sampler
# ---------------------------------------------------------------------------

def get_batch(data: torch.Tensor):
    """
    Sample BATCH_SIZE random (input, target) pairs from `data`.

    The target is the input shifted one position to the right:
        input  = data[i   : i+SEQ_LEN]
        target = data[i+1 : i+SEQ_LEN+1]
    so the model learns to predict the next token at every position.
    """
    # Random start indices; ensure at least SEQ_LEN+1 tokens remain after each.
    ix = torch.randint(len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    x  = torch.stack([data[i    : i + SEQ_LEN    ] for i in ix])  # inputs
    y  = torch.stack([data[i + 1: i + SEQ_LEN + 1] for i in ix])  # targets
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()   # decorator disables gradient tracking for the entire function
def evaluate(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor) -> dict:
    """
    Estimate average loss over EVAL_ITERS random batches from each split.

    Using multiple batches (rather than a single batch) gives a lower-variance
    loss estimate, which produces smoother and more reliable loss curves.

    Returns a dict with keys "train" and "val", each holding a float loss.
    """
    model.eval()   # disable dropout for deterministic evaluation

    results = {}
    splits  = {"train": train_data, "val": val_data}

    for split_name, split_data in splits.items():
        losses = torch.zeros(EVAL_ITERS)   # pre-allocate tensor to store each batch loss
        for k in range(EVAL_ITERS):
            x, y       = get_batch(split_data)
            _, loss    = model(x, y)       # forward pass only; no .backward()
            losses[k]  = loss.item()       # store scalar loss for this batch
        results[split_name] = losses.mean().item()  # average across all eval batches

    model.train()  # re-enable dropout before returning to the training loop
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Build dataset ---
    data, vocab_size, stoi, itos = build_toy_dataset()
    print(f"Corpus size : {len(data):,} tokens")
    print(f"Vocab size  : {vocab_size}  ({list(itos.values())})")

    # 90 / 10 train-validation split.
    # The split is a fixed index cut rather than a shuffle so that there is no
    # data leakage between the two sets.
    n          = int(0.9 * len(data))
    train_data = data[:n]   # first 90 % for learning
    val_data   = data[n:]   # last  10 % held out for evaluation
    print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}\n")

    # --- Build model ---
    model = GPT(
        vocab_size  = vocab_size,
        d_model     = D_MODEL,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        max_seq_len = SEQ_LEN,
        dropout     = DROPOUT,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}\n")

    # --- Optimiser ---
    # AdamW uses adaptive per-parameter learning rates (Adam) plus decoupled
    # weight decay which acts as L2 regularisation without interfering with
    # the adaptive moment estimates.
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-1)

    # --- Training loop ---
    print(f"{'Step':>6}  {'Train Loss':>10}  {'Val Loss':>10}")
    print("-" * 32)

    model.train()  # activate dropout

    for step in range(1, MAX_ITERS + 1):

        # Periodic evaluation — runs before the gradient step so that step 1
        # gives us the baseline (untrained) loss.
        if step % EVAL_EVERY == 0 or step == 1:
            metrics = evaluate(model, train_data, val_data)
            print(f"{step:>6}  {metrics['train']:>10.4f}  {metrics['val']:>10.4f}")
            model.train()  # evaluate() sets model.eval(); restore training mode

        # ----- Gradient step -----
        x, y = get_batch(train_data)        # sample a random mini-batch

        logits, loss = model(x, y)          # forward pass → cross-entropy loss

        optimiser.zero_grad()               # clear stale gradients from last step
        loss.backward()                     # backprop: compute ∂loss/∂θ for all θ

        # Clip gradients to prevent exploding updates in early training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()                    # θ ← θ − lr · grad

    # Final evaluation after all training steps.
    metrics = evaluate(model, train_data, val_data)
    print("-" * 32)
    print(f"{'Final':>6}  {metrics['train']:>10.4f}  {metrics['val']:>10.4f}")

    # --- Generate a sample ---
    # Feed the model a short prompt and let it complete the sequence.
    # This gives a qualitative feel for what the model has learned.
    print("\n--- Generated sample (prompt: '3+') ---")
    model.eval()

    prompt_str = "3+"                              # the prompt we condition on
    prompt_ids = [stoi[ch] for ch in prompt_str]  # encode each character
    prompt_tensor = torch.tensor(
        prompt_ids, dtype=torch.long, device=device
    ).unsqueeze(0)  # add batch dim: (T,) → (1, T)

    generated = model.generate(
        prompt_tensor,
        max_new_tokens = MAX_NEW_TOKENS,
        temperature    = 0.5,   # low temperature → more deterministic / confident output
        top_k          = 10,    # restrict sampling to the 10 most likely next tokens
    )

    # Decode token ids back to a string and print.
    print("".join(itos[i] for i in generated[0].tolist()))


if __name__ == "__main__":
    main()
