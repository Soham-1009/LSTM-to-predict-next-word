"""Tokenizer implementation for LSTM Language Model.

Supports special tokens:
- <PAD>: 0
- <UNK>: 1
- <BOS>: 2
- <EOS>: 3

Preserves meaningful punctuation and provides encode/decode methods with JSON serialization.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = SPECIAL_TOKENS

# Regex tokenizer pattern to capture words and punctuation marks separately
TOKEN_REGEX = re.compile(r"\w+|[^\w\s]")


class Tokenizer:
    """Word-level tokenizer with special tokens and vocabulary mapping."""

    def __init__(
        self,
        token_to_id: Optional[Dict[str, int]] = None,
        special_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        if token_to_id is not None:
            self.token_to_id = dict(token_to_id)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        else:
            # Initialize with special tokens
            self.token_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
            self.id_to_token = {idx: tok for idx, tok in enumerate(self.special_tokens)}

        self.pad_id = self.token_to_id.get(PAD_TOKEN, 0)
        self.unk_id = self.token_to_id.get(UNK_TOKEN, 1)
        self.bos_id = self.token_to_id.get(BOS_TOKEN, 2)
        self.eos_id = self.token_to_id.get(EOS_TOKEN, 3)

    def tokenize(self, text: str) -> List[str]:
        """Split text into word and punctuation tokens."""
        if not text:
            return []
        return TOKEN_REGEX.findall(text.strip().lower())

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """Convert input string to a list of integer token IDs."""
        tokens = self.tokenize(text)
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in tokens:
            tok_id = self.token_to_id.get(tok)
            if tok_id is None:
                tok_id = self.token_to_id.get(tok.lower(), self.unk_id)
            ids.append(tok_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = True
    ) -> str:
        """Convert a sequence of token IDs back into readable text.

        When skip_special_tokens is True, structural control tokens (<PAD>, <BOS>, <EOS>)
        are omitted from the output.
        """
        control_tokens = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
        tokens: List[str] = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if skip_special_tokens and tok in control_tokens:
                    continue
                tokens.append(tok)
            elif not skip_special_tokens:
                tokens.append(UNK_TOKEN)

        # Reconstruct sentence with proper spacing around punctuation
        output = ""
        for tok in tokens:
            if not output or tok in {".", ",", "!", "?", ";", ":", "'", ")", "]", "}"}:
                output += tok
            elif output.endswith("(") or output.endswith("["):
                output += tok
            else:
                output += " " + tok
        return output

    def build_vocab(
        self,
        token_frequencies: Dict[str, int],
        max_vocab_size: int = 30000,
        min_freq: int = 2,
    ) -> None:
        """Build vocabulary mapping from token frequency distribution."""
        # Start with special tokens
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.special_tokens)}

        # Sort non-special tokens by frequency descending
        sorted_tokens = sorted(
            token_frequencies.items(), key=lambda item: (-item[1], item[0])
        )

        for tok, freq in sorted_tokens:
            if tok in self.token_to_id:
                continue
            if freq < min_freq:
                break
            if len(self.token_to_id) >= max_vocab_size:
                break
            idx = len(self.token_to_id)
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

    def save(self, file_path: Path | str) -> None:
        """Persist tokenizer vocabulary to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: Path | str) -> "Tokenizer":
        """Load tokenizer vocabulary from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            token_to_id=payload["token_to_id"],
            special_tokens=payload.get("special_tokens"),
        )

    def __len__(self) -> int:
        return len(self.token_to_id)
