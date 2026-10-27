"""Controlled Domain-Adaptation Fine-Tuning Experiment.

Fine-tunes the baseline LSTM checkpoint (best_val.pt) on the multi-domain pilot dataset
to improve cross-domain generalization while strictly preserving the 30,000-token vocabulary
and architecture compatibility.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_project_root = Path(__file__).resolve().parent.parent.parent
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) in sys.path:
    sys.path.remove(str(_script_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.model.lstm import LSTMLanguageModel
from backend.model.tokenizer import Tokenizer
from backend.training.dataset import CausalLanguageModelDataset
from backend.training.evaluate import compute_top_k_accuracy


def run_domain_adaptation(
    baseline_checkpoint_path: Path | str = "backend/model_artifacts/checkpoints/best_val.pt",
    pilot_data_dir: Path | str = "experiments/multidomain_v1/pilot/processed",
    output_checkpoint_path: Path | str = "experiments/multidomain_v1/checkpoints/multidomain_adapted.pt",
    learning_rate: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 32,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute warm-start domain adaptation fine-tuning."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_path = Path(baseline_checkpoint_path)
    pilot_dir = Path(pilot_data_dir)
    output_path = Path(output_checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading baseline checkpoint from {baseline_path}...")
    checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]

    # Initialize model
    model = LSTMLanguageModel(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_idx=cfg["pad_idx"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Load datasets
    print(f"[2/5] Loading multi-domain pilot datasets from {pilot_dir}...")
    train_tokens = np.fromfile(pilot_dir / "train.bin", dtype=np.uint16)
    val_tokens = np.fromfile(pilot_dir / "val.bin", dtype=np.uint16)

    train_offsets = np.fromfile(pilot_dir / "train_offsets.bin", dtype=np.uint64)
    val_offsets = np.fromfile(pilot_dir / "val_offsets.bin", dtype=np.uint64)

    seq_len = cfg["context_length"]
    train_dataset = CausalLanguageModelDataset(
        token_data=train_tokens,
        seq_len=seq_len,
        stride=seq_len,
        doc_offsets=train_offsets,
    )
    val_dataset = CausalLanguageModelDataset(
        token_data=val_tokens,
        seq_len=seq_len,
        stride=seq_len,
        doc_offsets=val_offsets,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    print(f"  Train: {len(train_tokens):,} tokens ({len(train_dataset):,} sequences)")
    print(f"  Val:   {len(val_tokens):,} tokens ({len(val_dataset):,} sequences)")

    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["pad_idx"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initial validation evaluation before any fine-tuning
    print("\n[3/5] Evaluating pre-adaptation baseline on multi-domain validation set...")
    model.eval()
    init_val_loss = 0.0
    init_tokens = 0
    with torch.inference_mode():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            nv = int((y.view(-1) != cfg["pad_idx"]).sum().item())
            init_val_loss += loss.item() * nv
            init_tokens += nv
    baseline_val_loss = init_val_loss / max(init_tokens, 1)
    baseline_val_ppl = math.exp(min(baseline_val_loss, 20))
    print(f"  Baseline Pilot Val Loss: {baseline_val_loss:.4f} | PPL: {baseline_val_ppl:.2f}")

    # Training loop
    print(f"\n[4/5] Starting domain adaptation fine-tuning for {epochs} epochs (LR={learning_rate})...")
    best_val_loss = float("inf")
    epoch_records = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_tokens_count = 0
        ep_start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            nv = int((y.view(-1) != cfg["pad_idx"]).sum().item())
            total_train_loss += loss.item() * nv
            train_tokens_count += nv

        avg_train_loss = total_train_loss / max(train_tokens_count, 1)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_tokens_count = 0
        val_top1 = 0
        val_top5 = 0

        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)
                loss = criterion(logits_flat, y_flat)

                valid = y_flat != cfg["pad_idx"]
                nv = int(valid.sum().item())
                total_val_loss += loss.item() * nv
                val_tokens_count += nv

                _, top5_idx = torch.topk(logits_flat, 5, dim=-1)
                correct = top5_idx.eq(y_flat.unsqueeze(-1))
                val_top1 += int((correct[:, 0] & valid).sum().item())
                val_top5 += int((correct.any(dim=-1) & valid).sum().item())

        avg_val_loss = total_val_loss / max(val_tokens_count, 1)
        val_ppl = math.exp(min(avg_val_loss, 20))
        top1_pct = (val_top1 / max(val_tokens_count, 1)) * 100
        top5_pct = (val_top5 / max(val_tokens_count, 1)) * 100
        ep_duration = time.time() - ep_start

        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_perplexity": round(val_ppl, 2),
            "val_top1_accuracy": round(top1_pct, 2),
            "val_top5_accuracy": round(top5_pct, 2),
            "duration_seconds": round(ep_duration, 2),
        }
        epoch_records.append(record)
        print(
            f"  Epoch {epoch:2d}/{epochs:2d} ({ep_duration:.2f}s) | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"Val Top1: {top1_pct:.2f}% | "
            f"Val Top5: {top5_pct:.2f}%"
        )

        # Save checkpoint if best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "config": cfg,
                    "baseline_val_loss": baseline_val_loss,
                    "adaptation_learning_rate": learning_rate,
                },
                tmp_path,
            )
            os.replace(tmp_path, output_path)  # atomic on same filesystem

    total_training_time = time.time() - start_time
    print(f"\n[5/5] Domain adaptation complete in {total_training_time:.2f}s.")
    print(f"  Best Val Loss: {best_val_loss:.4f} (Baseline: {baseline_val_loss:.4f}, Delta: {best_val_loss - baseline_val_loss:.4f})")
    print(f"  Saved adapted checkpoint to: {output_path}")

    # Summary payload
    summary = {
        "experiment_name": "multidomain_v1_domain_adaptation",
        "baseline_checkpoint": str(baseline_path),
        "adapted_checkpoint": str(output_path),
        "device": str(device),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "baseline_pilot_val_loss": round(baseline_val_loss, 4),
        "baseline_pilot_val_ppl": round(baseline_val_ppl, 2),
        "adapted_pilot_val_loss": round(best_val_loss, 4),
        "adapted_pilot_val_ppl": round(math.exp(min(best_val_loss, 20)), 2),
        "total_training_seconds": round(total_training_time, 2),
        "epoch_records": epoch_records,
    }

    summary_file = output_path.parent / "adaptation_training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_domain_adaptation()
