"""PyTorch LSTM Language Model architecture for next-word prediction and text generation.

Matches the design specified in architecture.md:
- Embedding layer
- Multi-layer LSTM (default 2 layers, hidden size 512, embed size 256, dropout 0.3)
- Linear projection from hidden size to vocabulary logits
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLanguageModel(nn.Module):
    """LSTM-based Causal Language Model."""

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pad_idx = pad_idx

        # Layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through Embedding -> LSTM -> Dropout -> Linear Projection.

        Args:
            x: Tensor of token indices with shape (batch_size, seq_len)
            hidden: Optional tuple of (h_0, c_0) hidden states

        Returns:
            logits: Output tensor with shape (batch_size, seq_len, vocab_size)
            hidden: Tuple of updated (h_n, c_n) states
        """
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        dropped = self.dropout(lstm_out)
        logits = self.fc(dropped)  # (batch_size, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize zero hidden states for LSTM."""
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
        )


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False,
) -> int:
    """Sample the next token index given output logits of shape (vocab_size,).

    Supports greedy argmax, temperature scaling, top-k filtering, and nucleus (top-p) sampling.
    Safely handles non-finite, NaN, and zero-sum logits without crashing or leaking control tokens.
    """
    clean_logits = torch.nan_to_num(logits, nan=-float("inf"), posinf=1e9, neginf=-float("inf"))

    if greedy or temperature <= 0.0:
        if not torch.isfinite(clean_logits).any():
            return 1  # UNK fallback
        return int(torch.argmax(clean_logits).item())

    # Apply temperature
    clean_logits = clean_logits / max(temperature, 1e-5)

    # Top-K filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, clean_logits.size(-1))
        indices_to_remove = clean_logits < torch.topk(clean_logits, k)[0][..., -1, None]
        clean_logits[indices_to_remove] = -float("Inf")

    # Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(clean_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        clean_logits[indices_to_remove] = -float("Inf")

    probabilities = F.softmax(clean_logits, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    prob_sum = probabilities.sum().item()

    if prob_sum <= 0:
        if torch.isfinite(clean_logits).any():
            return int(torch.argmax(clean_logits).item())
        return 1

    probabilities = probabilities / prob_sum
    sampled_token = torch.multinomial(probabilities, num_samples=1).item()
    return int(sampled_token)
