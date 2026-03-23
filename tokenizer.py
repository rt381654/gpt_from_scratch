"""
tokenizer.py — Character-Level Tokenizer
=========================================
Converts raw text into integer token ids and back.

Character-level tokenisation is the simplest possible scheme:
  - Every unique character in the training corpus is one token.
  - vocab_size equals the number of distinct characters.
  - No external libraries or pre-trained vocabulary needed.
"""


class Tokenizer:
    """
    Builds a character-level vocabulary from a text string and provides
    encode / decode methods.
    """

    def __init__(self, text: str):
        """
        Args:
            text: The full corpus text used to build the vocabulary.
                  Every unique character in this string becomes a token.
        """
        # sorted() ensures the vocabulary order is deterministic across runs.
        # set() extracts the unique characters; sorted() gives a stable ordering.
        chars = sorted(set(text))

        # Total number of distinct tokens — used as vocab_size when building
        # the embedding table in the model.
        self.vocab_size = len(chars)

        # stoi (string-to-index): maps each character to a unique integer id.
        # enumerate(chars) yields (0, chars[0]), (1, chars[1]), …
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # itos (index-to-string): the reverse mapping, used when decoding
        # generated token ids back into human-readable text.
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of integer token ids."""
        # Look up each character in stoi; raises KeyError for unknown chars.
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        """Convert a sequence of integer token ids back into a string."""
        # Join the character for each id; ids can be a list or any iterable.
        return "".join(self.itos[i] for i in ids)
