"""PyTorch LSTM training pipeline runner.

Supports:
- Memory-mapped dataset loading (uint16/uint32)
- Multi-layer LSTM language modeling with causal shifted target
- AdamW optimizer with gradient clipping
- Validation evaluation (Cross-Entropy loss, perplexity, Top-1/Top-3/Top-5 accuracy)
- Checkpoint persistence (latest.pt and best_val.pt)
- Real-time GPU VRAM and tokens/sec throughput tracking
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path and remove script directory to avoid shadowing stdlib tokenize
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typing import Optional


import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from backend.model.config import ModelConfig, TrainingConfig
from backend.model.lstm import LSTMLanguageModel
from backend.model.tokenizer import Tokenizer
from backend.training.checkpoint import load_checkpoint, save_checkpoint
from backend.training.dataset import CausalLanguageModelDataset
from backend.training.evaluate import evaluate_model
from backend.training.serialize import load_memmap_tokens


def train_model(
    train_dataset: CausalLanguageModelDataset,
    val_dataset: Optional[CausalLanguageModelDataset] = None,
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    resume_checkpoint: Optional[Path | str] = None,
    lr_scheduler_type: Optional[str] = None,
    patience: Optional[int] = None,
) -> LSTMLanguageModel:
    """Train PyTorch LSTM Language Model with validation checkpointing and DDP support."""
    m_cfg = model_config or ModelConfig()
    t_cfg = train_config or TrainingConfig()

    # Detect DDP environment variables (set by torchrun)
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMLanguageModel(
        vocab_size=m_cfg.vocab_size,
        embed_dim=m_cfg.embed_dim,
        hidden_dim=m_cfg.hidden_dim,
        num_layers=m_cfg.num_layers,
        dropout=m_cfg.dropout,
        pad_idx=m_cfg.pad_idx,
    ).to(dev)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if is_ddp else model

    criterion = nn.CrossEntropyLoss(ignore_index=m_cfg.pad_idx)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=t_cfg.learning_rate,
        weight_decay=t_cfg.weight_decay,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    checkpoint_dir = Path(t_cfg.checkpoint_dir)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt = None
    if resume_checkpoint is not None:
        p = Path(resume_checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"Resume checkpoint file not found at: {resume_checkpoint}")
        if rank == 0:
            print(f"Resuming training from checkpoint: {resume_checkpoint}", flush=True)
        ckpt = load_checkpoint(resume_checkpoint, raw_model, optimizer, device=dev)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if rank == 0:
            print(f"Resumed at epoch {start_epoch} (best previous val loss: {best_val_loss:.4f})", flush=True)

    # Configure learning rate scheduler
    scheduler = None
    if lr_scheduler_type and lr_scheduler_type.lower() != "none":
        stype = lr_scheduler_type.lower()
        if stype == "cosine":
            for param_group in optimizer.param_groups:
                param_group["lr"] = t_cfg.learning_rate
            total_remaining = max(t_cfg.epochs - start_epoch + 1, 1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_remaining, eta_min=1e-5)
            if rank == 0:
                print(f"Configured CosineAnnealingLR: T_max={total_remaining}, eta_min=1e-5 (Initial LR: {t_cfg.learning_rate})", flush=True)
        elif stype == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5)
            if rank == 0:
                print("Configured ReduceLROnPlateau: factor=0.5, patience=1, min_lr=1e-5", flush=True)
        elif stype in ("step", "exponential"):
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
            if rank == 0:
                print("Configured ExponentialLR: gamma=0.85", flush=True)

    # DataLoaders: configure DistributedSampler under DDP
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = (
            DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if val_dataset is not None
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=t_cfg.batch_size,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=(dev.type == "cuda"),
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=t_cfg.batch_size,
                sampler=val_sampler,
                shuffle=False,
                pin_memory=(dev.type == "cuda"),
            )
            if val_dataset is not None
            else None
        )
    else:
        train_sampler = None
        val_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=t_cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(dev.type == "cuda"),
        )
        val_loader = (
            DataLoader(val_dataset, batch_size=t_cfg.batch_size, shuffle=False, pin_memory=(dev.type == "cuda"))
            if val_dataset is not None
            else None
        )

    effective_global_batch = t_cfg.batch_size * world_size * getattr(t_cfg, "gradient_accumulation_steps", 1)
    if rank == 0:
        total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print("=" * 70, flush=True)
        print(f"Starting LSTM Training on Device: {dev} (DDP={is_ddp}, World Size={world_size})", flush=True)
        if dev.type == "cuda":
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
            print(f"Allocated VRAM (Rank 0): {torch.cuda.memory_allocated(dev) / (1024**2):.1f} MB", flush=True)
        print(f"Model Parameters: {total_params:,}", flush=True)
        print(f"Distributed Workers (world_size): {world_size}", flush=True)
        print(f"Per-Process / Per-GPU Batch Size: {t_cfg.batch_size}", flush=True)
        print(f"Gradient Accumulation Steps: {getattr(t_cfg, 'gradient_accumulation_steps', 1)}", flush=True)
        print(f"Global Effective Batch Size: {effective_global_batch}", flush=True)
        if is_ddp and world_size > 1 and t_cfg.batch_size >= 64 and getattr(t_cfg, "global_batch_size", None) != effective_global_batch:
            print(f"NOTE: DDP active with {world_size} GPUs and per-GPU batch_size={t_cfg.batch_size}, yielding global batch size of {effective_global_batch}. For Kaggle 2-GPU baseline with global batch size 64, use per-GPU batch_size=32.", flush=True)
        print(f"Context Length: {m_cfg.context_length}", flush=True)
        print(f"Epochs: {t_cfg.epochs} | Learning Rate: {t_cfg.learning_rate}", flush=True)
        print("=" * 70, flush=True)


    epochs_no_improve = 0

    for epoch in range(start_epoch, t_cfg.epochs + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        total_tokens = 0
        epoch_start_time = time.time()
        step_start_time = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()

            logits, _ = model(x)
            loss = criterion(logits.view(-1, m_cfg.vocab_size), y.view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=t_cfg.grad_clip)
            optimizer.step()

            batch_tokens = x.numel()
            running_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            # Periodic status log (rank 0 only)
            if rank == 0 and ((batch_idx + 1) % 100 == 0 or (max_train_batches and (batch_idx + 1) == max_train_batches)):
                step_elapsed = max(time.time() - step_start_time, 1e-4)
                effective_batch_tokens = batch_tokens * world_size
                tokens_sec = (100 * effective_batch_tokens) / step_elapsed if (batch_idx + 1) % 100 == 0 else effective_batch_tokens / step_elapsed
                step_loss = loss.item()
                step_ppl = math.exp(min(step_loss, 20))
                vram_info = ""
                if dev.type == "cuda":
                    vram_mb = torch.cuda.memory_allocated(dev) / (1024**2)
                    vram_peak = torch.cuda.max_memory_allocated(dev) / (1024**2)
                    vram_info = f" | VRAM (Rank 0): {vram_mb:.0f}MB (Peak: {vram_peak:.0f}MB)"

                print(
                    f"Epoch {epoch}/{t_cfg.epochs} [{batch_idx+1:,} batches] "
                    f"Loss: {step_loss:.4f} | PPL: {step_ppl:.2f} | Speed: {tokens_sec:,.0f} tok/s{vram_info}",
                    flush=True,
                )
                step_start_time = time.time()

        # Aggregate training loss and token count across ranks for accurate epoch loss
        if is_ddp and dist.is_initialized():
            t_stats = torch.tensor([running_loss, float(total_tokens)], device=dev, dtype=torch.float64)
            dist.all_reduce(t_stats, op=dist.ReduceOp.SUM)
            global_train_loss = t_stats[0].item()
            global_total_tokens = int(t_stats[1].item())
        else:
            global_train_loss = running_loss
            global_total_tokens = total_tokens

        avg_train_loss = global_train_loss / max(global_total_tokens, 1)
        train_ppl = math.exp(min(avg_train_loss, 20))
        epoch_time = time.time() - epoch_start_time
        if rank == 0:
            print(
                f"\n--- Epoch {epoch}/{t_cfg.epochs} Complete ({epoch_time:.1f}s) "
                f"Avg Train Loss: {avg_train_loss:.4f} | PPL: {train_ppl:.2f} ---",
                flush=True,
            )

        # Validation phase
        val_loss = None
        val_time = 0.0
        metrics = None
        if val_loader is not None:
            if rank == 0:
                print("Evaluating on validation split...", flush=True)
            val_start = time.time()
            metrics = evaluate_model(model, val_loader, criterion, dev, max_batches=max_val_batches)
            val_time = time.time() - val_start
            val_loss = metrics["loss"]
            val_ppl = metrics["perplexity"]
            if rank == 0:
                print(
                    f"Validation Results: Loss: {val_loss:.4f} | PPL: {val_ppl:.2f} | "
                    f"Top-1: {metrics['top1_acc']*100:.2f}% | Top-3: {metrics['top3_acc']*100:.2f}% | "
                    f"Top-5: {metrics['top5_acc']*100:.2f}% (Time: {val_time:.2f}s)",
                    flush=True,
                )

            # Step learning rate scheduler if configured (all ranks to maintain synchronous LR)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                if rank == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f"Learning rate updated to: {current_lr:.6e}", flush=True)

            # Save latest checkpoint (rank 0 only, saving raw_model without 'module.' prefix)
            if rank == 0:
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_loss=val_loss,
                    config=m_cfg.to_dict(),
                )

            # Save best validation checkpoint and track patience (all ranks track metric)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if rank == 0:
                    save_checkpoint(
                        checkpoint_dir / "best_val.pt",
                        model=raw_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        val_loss=val_loss,
                        config=m_cfg.to_dict(),
                    )
                    print(f" -> Best validation model saved to {checkpoint_dir / 'best_val.pt'} (Loss: {val_loss:.4f})", flush=True)
            else:
                epochs_no_improve += 1
                if rank == 0 and patience is not None:
                    print(f"Validation loss did not improve ({epochs_no_improve}/{patience} patience epochs).", flush=True)

        # Early stopping check
        stop_training = False
        if patience is not None and epochs_no_improve >= patience:
            stop_training = True
            if rank == 0:
                print(f"Early stopping triggered after {patience} epochs without validation improvement.", flush=True)

        if is_ddp and dist.is_initialized():
            stop_tensor = torch.tensor(1 if stop_training else 0, device=dev)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item() == 1)

        if stop_training:
            break

        # Benchmark Summary Report (rank 0 only)
        if rank == 0:
            avg_tokens_sec = global_total_tokens / max(epoch_time, 1e-4)
            print("\n" + "=" * 70, flush=True)
            print("BENCHMARK EXECUTION SUMMARY", flush=True)
            print("=" * 70, flush=True)
            print(f"Epoch:           {epoch}", flush=True)
            print(f"Train loss:      {avg_train_loss:.4f} (PPL: {train_ppl:.2f})", flush=True)
            if val_loss is not None and metrics is not None:
                print(f"Validation loss: {val_loss:.4f} (PPL: {val_ppl:.2f})", flush=True)
                print(f"Top-1 accuracy:  {metrics['top1_acc']*100:.2f}%", flush=True)
                print(f"Top-3 accuracy:  {metrics['top3_acc']*100:.2f}%", flush=True)
                print(f"Top-5 accuracy:  {metrics['top5_acc']*100:.2f}%", flush=True)
            print(f"Training time:   {epoch_time:.2f}s ({epoch_time/60:.2f} min)", flush=True)
            print(f"Validation time: {val_time:.2f}s ({val_time/60:.2f} min)", flush=True)
            print(f"Tokens/sec:      {avg_tokens_sec:,.0f} tokens/s", flush=True)
            if dev.type == "cuda":
                vram_alloc = torch.cuda.memory_allocated(dev) / (1024**2)
                vram_peak = torch.cuda.max_memory_allocated(dev) / (1024**2)
                vram_total = torch.cuda.get_device_properties(dev).total_memory / (1024**2)
                print(f"GPU memory usage: {vram_alloc:.1f} MB allocated, {vram_peak:.1f} MB peak (out of {vram_total:.0f} MB total VRAM)", flush=True)
            else:
                print("GPU memory usage: N/A (CPU execution)", flush=True)
            print("=" * 70, flush=True)

    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

    return raw_model



def main():
    parser = argparse.ArgumentParser(
        description="Train PyTorch LSTM Language Model on preprocessed binary corpus."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="backend/datasets/processed",
        help="Directory containing train.bin and val.bin",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="backend/model_artifacts/tokenizer/vocab.json",
        help="Path to saved vocab.json artifact",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="backend/model_artifacts/checkpoints",
        help="Directory to save checkpoints (latest.pt, best_val.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size per process/GPU (default: 64; use 32 on 2-GPU DDP to keep global batch 64)",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Target effective global batch size across all GPUs (e.g. 64). When specified, automatically derives per-GPU batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay parameter",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=35,
        help="Context sequence length (T)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Word embedding dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="LSTM hidden units per layer",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of stacked LSTM layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
        help="Dropout probability",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional limit on number of train batches per epoch (for smoke testing / benchmarking)",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional limit on number of validation batches",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sequence stride (default: context_length for non-overlapping causal LM sequences)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Hardware compute device (default: auto-detect CUDA if available)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "plateau", "step", "exponential"],
        help="Learning rate decay schedule (default: none)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Validation early stopping patience (epochs without improvement)",
    )

    args = parser.parse_args()

    # Load tokenizer to determine actual vocabulary size
    vocab_file = Path(args.vocab_path)
    if not vocab_file.exists():
        print(f"Error: Vocabulary file not found at {vocab_file}. Run data pipeline first.")
        sys.exit(1)

    tokenizer = Tokenizer.load(vocab_file)
    actual_vocab_size = len(tokenizer)

    # Determine dtype from metadata or vocab size
    dtype = np.uint16 if actual_vocab_size <= 65535 else np.uint32

    # Sequence stride: defaults to context length for standard language modeling
    stride = args.stride if args.stride is not None else args.context_length

    # Load memory-mapped datasets
    data_path = Path(args.data_dir)
    train_bin = data_path / "train.bin"
    val_bin = data_path / "val.bin"

    is_rank0 = int(os.environ.get("RANK", 0)) == 0

    if not train_bin.exists():
        if is_rank0:
            print(f"Error: Training binary not found at {train_bin}. Run data pipeline first.")
        sys.exit(1)

    if is_rank0:
        print(f"Loading memory-mapped training data from {train_bin} (dtype={dtype.__name__}, stride={stride})...")
    train_offsets_bin = data_path / "train_offsets.bin"
    train_offsets = np.fromfile(train_offsets_bin, dtype=np.uint64) if train_offsets_bin.exists() else None
    train_tokens = load_memmap_tokens(train_bin, dtype=dtype)
    train_dataset = CausalLanguageModelDataset(train_tokens, seq_len=args.context_length, stride=stride, doc_offsets=train_offsets)
    if is_rank0:
        print(f"Train dataset loaded: {len(train_tokens):,} total tokens, {len(train_dataset):,} causal sequences ({len(train_dataset)//args.batch_size:,} batches).")

    val_dataset = None
    if val_bin.exists():
        if is_rank0:
            print(f"Loading memory-mapped validation data from {val_bin} (stride={stride})...")
        val_offsets_bin = data_path / "val_offsets.bin"
        val_offsets = np.fromfile(val_offsets_bin, dtype=np.uint64) if val_offsets_bin.exists() else None
        val_tokens = load_memmap_tokens(val_bin, dtype=dtype)
        val_dataset = CausalLanguageModelDataset(val_tokens, seq_len=args.context_length, stride=stride, doc_offsets=val_offsets)
        if is_rank0:
            print(f"Val dataset loaded: {len(val_tokens):,} total tokens, {len(val_dataset):,} sequences ({len(val_dataset)//args.batch_size:,} batches).")

    m_cfg = ModelConfig(
        vocab_size=actual_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        context_length=args.context_length,
        pad_idx=tokenizer.pad_id,
        unk_idx=tokenizer.unk_id,
        bos_idx=tokenizer.bos_id,
        eos_idx=tokenizer.eos_id,
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = max(args.gradient_accumulation_steps, 1)

    effective_batch_size = args.batch_size
    if args.global_batch_size is not None:
        divisor = world_size * grad_accum
        if args.global_batch_size % divisor != 0:
            raise ValueError(
                f"--global-batch-size ({args.global_batch_size}) must be divisible by "
                f"world_size ({world_size}) * gradient_accumulation_steps ({grad_accum}) = {divisor}."
            )
        computed_batch = args.global_batch_size // divisor
        if args.batch_size != 64 and args.batch_size != computed_batch:
            raise ValueError(
                f"Conflicting batch size parameters: --batch-size={args.batch_size} vs "
                f"--global-batch-size={args.global_batch_size} (which implies per-GPU batch size {computed_batch} "
                f"across world_size={world_size}). Please specify only one or ensure they match."
            )
        effective_batch_size = computed_batch

    t_cfg = TrainingConfig(
        batch_size=effective_batch_size,
        global_batch_size=args.global_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )


    dev = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_config=m_cfg,
        train_config=t_cfg,
        device=dev,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume_checkpoint=args.resume,
        lr_scheduler_type=args.lr_scheduler,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
