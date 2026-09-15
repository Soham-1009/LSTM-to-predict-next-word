"""PyTorch LSTM model for next-word prediction.

The architecture is intentionally simple and academic:

    Embedding  ->  LSTM  ->  Dropout  ->  Linear  ->  vocabulary logits

Only the last time-step's hidden state is used to project onto the vocabulary.
This keeps the model tractable on a CPU and well suited to a small text corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    """Container for architecture hyperparameters.

    Persisting this dataclass as JSON in `config.json` is what allows the
    backend to reconstruct the exact same model that was trained.
    """

    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 5
    pad_idx: int = 0

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "sequence_length": self.sequence_length,
            "pad_idx": self.pad_idx,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LSTMConfig":
        return cls(
            vocab_size=int(data["vocab_size"]),
            embedding_dim=int(data.get("embedding_dim", 128)),
            hidden_dim=int(data.get("hidden_dim", 256)),
            num_layers=int(data.get("num_layers", 2)),
            dropout=float(data.get("dropout", 0.2)),
            sequence_length=int(data.get("sequence_length", 5)),
            pad_idx=int(data.get("pad_idx", 0)),
        )


class LSTMPredictor(nn.Module):
    """LSTM-based next-word prediction model.

    Inputs are token-id sequences of shape (batch, sequence_length).
    Outputs are vocabulary logits of shape (batch, vocab_size) for the last
    time-step only, because the next-word task only needs the final hidden
    state.
    """

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

        self._init_weights()

    # ------------------------------------------------------------------ utils
    def _init_weights(self) -> None:
        """Initialize weights to small, sensible values for stable training."""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run a forward pass.

        Args:
            input_ids: Tensor of shape (batch, sequence_length) with token ids.
            hidden: Optional pre-existing (h, c) state.

        Returns:
            logits: Tensor of shape (batch, vocab_size) for the last time-step.
            hidden: Updated (h, c) state.
        """
        embedded = self.embedding(input_ids)            # (B, T, E)
        output, hidden = self.lstm(embedded, hidden)    # (B, T, H), ((L,B,H),(L,B,H))
        last = output[:, -1, :]                         # (B, H)
        last = self.dropout(last)
        logits = self.fc(last)                          # (B, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate an initial hidden state on the given device."""
        h = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
        c = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
        return h, c
