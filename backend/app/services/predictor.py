"""High-level inference services used by the FastAPI layer.

These functions wrap the loaded PyTorch model and tokenizer, handle tensor
shaping, and return plain Python types so the API layer stays simple.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..models.lstm_model import LSTMPredictor
from ..utils.text_processing import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    Tokenizer,
    normalize_text,
    tokenize,
)
from .model_loader import model_manager

logger = logging.getLogger(__name__)

# Tokens that should never be surfaced to the user as a "next word".
_BLOCKED_TOKENS = {PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _prepare_input_sequence(
    text: str,
    tokenizer: Tokenizer,
    sequence_length: int,
) -> Tuple[List[int], List[str]]:
    """Normalize + encode the user's text and pad/truncate to sequence_length.

    Returns (token_ids, original_tokens). The token list is what the
    generator uses to append new words.
    """
    cleaned = normalize_text(text)
    tokens = tokenize(cleaned)
    if not tokens:
        return [], []

    token_ids = tokenizer.encode(tokens)
    if len(token_ids) < sequence_length:
        # Left-pad with <PAD> so the most recent tokens feed into the LSTM.
        token_ids = [tokenizer.pad_idx] * (sequence_length - len(token_ids)) + token_ids
    else:
        # Keep the most recent `sequence_length` tokens.
        token_ids = token_ids[-sequence_length:]
    return token_ids, tokens


def _topk_from_logits(
    logits: torch.Tensor,
    tokenizer: Tokenizer,
    top_k: int,
    temperature: float = 1.0,
) -> List[Tuple[str, float]]:
    """Return (word, probability) pairs sorted by probability descending."""
    if temperature <= 0:
        temperature = 1.0
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    k = min(top_k, probs.size(-1))
    top_probs, top_indices = torch.topk(probs, k=k)
    out: List[Tuple[str, float]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        word = tokenizer.idx2word.get(int(idx), UNK_TOKEN)
        # Hide the special tokens from predictions — they are not meaningful
        # next words to surface to the user.
        if word in _BLOCKED_TOKENS:
            continue
        out.append((word, float(prob)))
        if len(out) == top_k:
            break
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    word: str
    probability: float


def predict_next_word(
    text: str,
    top_k: int = 5,
    temperature: float = 1.0,
) -> List[Prediction]:
    """Predict the top-k most likely next words for `text`."""
    loaded = model_manager.get()
    model: LSTMPredictor = loaded.model
    tokenizer: Tokenizer = loaded.tokenizer
    device: torch.device = loaded.device
    sequence_length: int = loaded.config.sequence_length

    token_ids, _ = _prepare_input_sequence(text, tokenizer, sequence_length)
    if not token_ids:
        return []

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits, _ = model(input_tensor)

    pairs = _topk_from_logits(logits[0], tokenizer, top_k, temperature=temperature)
    return [Prediction(word=w, probability=p) for w, p in pairs]


def generate_text(
    text: str,
    num_words: int = 5,
    temperature: float = 1.0,
    strategy: str = "greedy",
) -> List[str]:
    """Autoregressively generate `num_words` continuation tokens.

    Returns the generated words (without the original seed).
    """
    loaded = model_manager.get()
    model: LSTMPredictor = loaded.model
    tokenizer: Tokenizer = loaded.tokenizer
    device: torch.device = loaded.device
    sequence_length: int = loaded.config.sequence_length

    seed_ids, _ = _prepare_input_sequence(text, tokenizer, sequence_length)
    if not seed_ids:
        return []

    generated: List[int] = []
    current_ids: List[int] = list(seed_ids)

    with torch.inference_mode():
        for _ in range(num_words):
            input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
            logits, _ = model(input_tensor)
            next_id = _sample_next_token(logits[0], tokenizer, temperature, strategy)
            if next_id is None:
                break
            word = tokenizer.idx2word.get(int(next_id), UNK_TOKEN)
            if word in _BLOCKED_TOKENS:
                # If the model wants to emit a special token, fall back to the
                # top non-special token to keep the generation readable.
                pairs = _topk_from_logits(logits[0], tokenizer, top_k=1, temperature=temperature)
                if not pairs:
                    break
                word = pairs[0][0]
                next_id = tokenizer.word2idx.get(word)
            generated.append(int(next_id))
            current_ids = current_ids[1:] + [int(next_id)]

    return tokenizer.decode(generated)


def _sample_next_token(
    logits: torch.Tensor,
    tokenizer: Tokenizer,
    temperature: float,
    strategy: str,
) -> int | None:
    """Sample a single next-token id using the requested strategy."""
    if strategy == "greedy" or temperature <= 0.1:
        return int(torch.argmax(logits, dim=-1).item())

    scaled = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled, dim=-1)

    k = max(1, min(10, probs.size(-1)))
    top_probs, top_indices = torch.topk(probs, k=k)
    top_probs = top_probs / top_probs.sum()

    choice = random.choices(top_indices.tolist(), weights=top_probs.tolist(), k=1)
    return int(choice[0])


# ---------------------------------------------------------------------------
# Sanity helpers for the API layer
# ---------------------------------------------------------------------------
def is_input_too_short(text: str, sequence_length: int) -> bool:
    """Return True if the cleaned token list has <1 token to feed the model."""
    return len(tokenize(normalize_text(text))) < 1


def model_metadata() -> dict:
    """Return metadata describing the loaded model, for /model-info."""
    loaded = model_manager.get()
    cfg = loaded.config
    return {
        "architecture": "LSTM",
        "framework": "PyTorch",
        "vocab_size": cfg.vocab_size,
        "embedding_dim": cfg.embedding_dim,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "sequence_length": cfg.sequence_length,
        "device": loaded.device.type,
        "parameter_count": loaded.model.num_parameters(),
    }
