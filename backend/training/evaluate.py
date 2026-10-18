"""Evaluation metrics for LSTM Language Model: loss, perplexity, and Top-K accuracy."""

import math
from typing import Dict, Optional
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1,
    pad_idx: Optional[int] = None,
) -> float:
    """Compute Top-K accuracy over flattened token predictions."""
    if targets.numel() == 0 or logits.numel() == 0:
        return 0.0
    # logits: (N, vocab_size), targets: (N)
    k = min(k, logits.size(-1))
    _, top_k_indices = torch.topk(logits, k, dim=-1)
    correct = top_k_indices.eq(targets.unsqueeze(-1).expand_as(top_k_indices)).any(dim=-1)
    if pad_idx is not None:
        valid_mask = targets != pad_idx
        num_valid = int(valid_mask.sum().item())
        if num_valid == 0:
            return 0.0
        return float((correct & valid_mask).float().sum().item() / num_valid)
    return float(correct.float().mean().item())


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
    pad_idx: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate model on a DataLoader and compute PRD-specified metrics (supporting DDP aggregation)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    effective_pad_idx = pad_idx if pad_idx is not None else getattr(criterion, "ignore_index", 0)

    is_rank0 = True
    if dist.is_available() and dist.is_initialized():
        is_rank0 = dist.get_rank() == 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        logits, _ = model(x)
        # Reshape to (batch_size * seq_len, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.view(-1)

        # Mask out padding tokens
        valid_mask = targets_flat != effective_pad_idx
        num_valid = int(valid_mask.sum().item())
        if num_valid > 0:
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item() * num_valid
            total_tokens += num_valid

            # Calculate Top-1, 3, 5 accuracy efficiently over non-padding tokens
            _, top5_indices = torch.topk(logits_flat, 5, dim=-1)
            targets_exp = targets_flat.unsqueeze(-1)
            correct_top5 = top5_indices.eq(targets_exp)
            top1_correct += int((correct_top5[:, 0] & valid_mask).sum().item())
            top3_correct += int((correct_top5[:, :3].any(dim=-1) & valid_mask).sum().item())
            top5_correct += int((correct_top5.any(dim=-1) & valid_mask).sum().item())

        if is_rank0 and (batch_idx + 1) % 500 == 0:
            print(f"Validation progress: [{batch_idx+1:,} batches evaluated]", flush=True)

    # In DDP, aggregate stats across all ranks via NCCL all_reduce
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor(
            [total_loss, float(total_tokens), float(top1_correct), float(top3_correct), float(top5_correct)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        total_tokens = int(stats[1].item())
        top1_correct = stats[2].item()
        top3_correct = stats[3].item()
        top5_correct = stats[4].item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "top1_acc": top1_correct / max(total_tokens, 1),
        "top3_acc": top3_correct / max(total_tokens, 1),
        "top5_acc": top5_correct / max(total_tokens, 1),
    }

