"""Evaluation utilities: loss, accuracy, perplexity on a DataLoader."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.models.lstm_model import LSTMPredictor


@torch.inference_mode()
def evaluate(
    model: LSTMPredictor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Return (average_loss, accuracy, perplexity) over the loader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == targets).sum().item()
        total_count += inputs.size(0)

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, accuracy, perplexity
