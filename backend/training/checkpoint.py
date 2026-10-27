"""Checkpoint management for training resumption and model artifact deployment."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    file_path: Path | str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    val_loss: float = float("inf"),
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model state dict, optimizer state, scheduler state, and training metadata.

    Writes to a temporary file first, then atomically replaces the target to
    prevent corrupted checkpoints on crash.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on same filesystem


def load_checkpoint(
    file_path: Path | str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model weights and optional optimizer and scheduler state from checkpoint."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {file_path}")

    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
