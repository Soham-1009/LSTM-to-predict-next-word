"""Reusable text preprocessing and tokenizer utilities.

The tokenizer is the same one persisted to disk by `training/train.py`. Keeping
preprocessing logic here lets both the training script and the inference layer
share an identical normalization pipeline, which is critical for matching the
model's expectations at runtime.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence

import pickle


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
BOS_TOKEN: str = "<BOS>"
EOS_TOKEN: str = "<EOS>"

SPECIAL_TOKENS: List[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
_CONTRACTIONS = {
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

_PUNCT_RE = re.compile(r"([.,!?;:\"'])")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[^\x20-\x7E]")


def normalize_text(text: str) -> str:
    """Normalize a raw text string for the corpus.

    Steps: Unicode normalize -> strip non-printables -> expand contractions ->
    isolate punctuation with spaces -> collapse whitespace -> lowercase.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _NON_PRINTABLE_RE.sub(" ", text)
    for short, expanded in _CONTRACTIONS.items():
        text = text.replace(short, expanded)
    text = _PUNCT_RE.sub(r" \1 ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip().lower()


def tokenize(text: str) -> List[str]:
    """Whitespace tokenizer operating on already-normalized text."""
    if not text:
        return []
    return [tok for tok in text.split() if tok]


# ---------------------------------------------------------------------------
# Vocabulary / Tokenizer
# ---------------------------------------------------------------------------
class Tokenizer:
    """Maps words to integer ids and back.

    A single instance is pickled to disk and reused both at training time and
    at inference time, guaranteeing consistent indexing.
    """

    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._frozen: bool = False

    # ----- building --------------------------------------------------------
    def build_from_corpus(self, token_sequences: Iterable[Sequence[str]], min_freq: int = 2) -> None:
        """Build the vocabulary from an iterable of already-tokenized sentences."""
        counter: Counter[str] = Counter()
        for seq in token_sequences:
            counter.update(seq)

        # Special tokens first so their indices are stable.
        self.word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for word, freq in counter.most_common():
            if freq < min_freq:
                continue
            if word in self.word2idx:
                continue
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._frozen = True

    # ----- encoding --------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[UNK_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.word2idx[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[EOS_TOKEN]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        """Convert tokens to ids, mapping unknown words to <UNK>."""
        unk = self.unk_idx
        return [self.word2idx.get(tok, unk) for tok in tokens]

    def encode_text(self, text: str) -> List[int]:
        """Normalize -> tokenize -> encode a raw string."""
        return self.encode(tokenize(normalize_text(text)))

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.idx2word.get(int(i), UNK_TOKEN) for i in ids]

    def decode_text(self, ids: Sequence[int]) -> str:
        return " ".join(self.decode(ids))

    # ----- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {"word2idx": self.word2idx, "idx2word": self.idx2word},
                fh,
            )

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        tokenizer = cls()
        tokenizer.word2idx = dict(payload["word2idx"])
        tokenizer.idx2word = {int(i): w for i, w in payload["idx2word"].items()}
        tokenizer._frozen = True
        return tokenizer


# ---------------------------------------------------------------------------
# Sliding window helpers (used by training, exposed for tests).
# ---------------------------------------------------------------------------
def build_sliding_windows(
    token_ids: Sequence[int],
    sequence_length: int,
) -> tuple[List[List[int]], List[int]]:
    """Convert a flat token-id stream into (inputs, targets) for next-word LM.

    For each index i such that i + sequence_length < len(token_ids), input is
    token_ids[i : i + sequence_length] and target is token_ids[i + sequence_length].
    """
    inputs: List[List[int]] = []
    targets: List[int] = []
    limit = len(token_ids) - sequence_length
    for i in range(limit):
        inputs.append(list(token_ids[i : i + sequence_length]))
        targets.append(int(token_ids[i + sequence_length]))
    return inputs, targets
