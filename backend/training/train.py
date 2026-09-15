"""End-to-end training pipeline for the LSTM next-word model.

Run from the `backend/` directory:

    python -m training.train --epochs 20

Artifacts produced under `backend/model/`:

    - lstm_model.pth        -- trained state_dict
    - tokenizer.pkl         -- fitted Tokenizer
    - config.json           -- architecture & vocabulary configuration
    - training_history.json -- per-epoch loss/accuracy/perplexity
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow `python training/train.py` and `python -m training.train` alike.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.lstm_model import LSTMConfig, LSTMPredictor  # noqa: E402
from app.utils.text_processing import (  # noqa: E402
    SPECIAL_TOKENS,
    Tokenizer,
    build_sliding_windows,
)
from training.dataset import make_dataloaders  # noqa: E402
from training.evaluate import evaluate  # noqa: E402
from training.preprocess import (  # noqa: E402
    DATASETS_DIR,
    GUTENBERG_BOOKS,
    corpus_stats,
    load_bundled_corpus,
    load_corpus_from_file,
    load_gutenberg_corpus,
    preprocess_corpus,
)

logger = logging.getLogger("train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI / configuration
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the LSTM next-word model")
    p.add_argument(
        "--corpus",
        type=str,
        default="bundled",
        choices=["bundled", "gutenberg", "file"],
        help="Which corpus to train on. 'bundled' is the small in-repo sample; "
             "'gutenberg' downloads a public-domain book; 'file' reads from "
             "--corpus-path.",
    )
    p.add_argument(
        "--corpus-name",
        type=str,
        default="alice",
        help="For --corpus gutenberg, the book key. "
             f"Available: {', '.join(sorted(GUTENBERG_BOOKS))}.",
    )
    p.add_argument(
        "--corpus-path",
        type=str,
        default="",
        help="For --corpus file, path to a plain-text corpus.",
    )
    p.add_argument(
        "--redownload",
        action="store_true",
        help="Force re-download of a Gutenberg corpus even if cached.",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--sequence-length", type=int, default=5)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-vocab", type=int, default=10000)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--lr-scheduler",
        type=str,
        default="plateau",
        choices=["none", "plateau"],
        help="Reduce learning rate on validation-loss plateau.",
    )
    p.add_argument("--lr-factor", type=float, default=0.5, help="LR shrink factor for the plateau scheduler.")
    p.add_argument("--lr-patience", type=int, default=2, help="Plateau scheduler patience (epochs).")
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training if val loss doesn't improve for this many epochs. 0 disables.",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--save-config-name",
        type=str,
        default="config.json",
        help="Filename inside model/ for the architecture config.",
    )
    p.add_argument(
        "--save-model-name",
        type=str,
        default="lstm_model.pth",
        help="Filename inside model/ for the trained weights.",
    )
    p.add_argument(
        "--save-tokenizer-name",
        type=str,
        default="tokenizer.pkl",
        help="Filename inside model/ for the tokenizer.",
    )
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tokenizer building (with optional vocab cap)
# ---------------------------------------------------------------------------
def _build_tokenizer(
    token_sequences: List[List[str]], min_freq: int, max_vocab: int
) -> Tokenizer:
    tokenizer = Tokenizer()
    tokenizer.build_from_corpus(token_sequences, min_freq=min_freq)
    if max_vocab and tokenizer.vocab_size > max_vocab:
        # Trim to top-N most frequent + specials.  We re-count manually here
        # so the cap is respected even if build_from_corpus kept everything.
        from collections import Counter

        counter: Counter[str] = Counter()
        for seq in token_sequences:
            counter.update(seq)
        keep = [w for w, _ in counter.most_common(max_vocab - len(SPECIAL_TOKENS))]
        tokenizer.word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for w in keep:
            if w in tokenizer.word2idx:
                continue
            tokenizer.word2idx[w] = len(tokenizer.word2idx)
        tokenizer.idx2word = {i: w for w, i in tokenizer.word2idx.items()}
        tokenizer._frozen = True
    return tokenizer


# ---------------------------------------------------------------------------
# Corpus dispatch
# ---------------------------------------------------------------------------
def _load_corpus(args: argparse.Namespace) -> Tuple[str, str]:
    """Resolve the (raw_text, source_label) pair from CLI args.

    The source_label is included in the training log and saved into
    training_history.json so you can always tell what the model was
    trained on.
    """
    if args.corpus == "bundled":
        return load_bundled_corpus(), "bundled sample corpus"
    if args.corpus == "gutenberg":
        info = GUTENBERG_BOOKS.get(args.corpus_name.lower())
        if info is None:
            raise SystemExit(
                f"Unknown Gutenberg book '{args.corpus_name}'. "
                f"Available: {', '.join(sorted(GUTENBERG_BOOKS))}."
            )
        text = load_gutenberg_corpus(args.corpus_name, force=args.redownload)
        label = f"gutenberg:{args.corpus_name} ({info['title']} by {info['author']})"
        return text, label
    if args.corpus == "file":
        if not args.corpus_path:
            raise SystemExit("--corpus file requires --corpus-path to be set.")
        return load_corpus_from_file(args.corpus_path), f"file:{args.corpus_path}"
    raise SystemExit(f"Unsupported corpus choice: {args.corpus}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _train_one_epoch(
    model: LSTMPredictor,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == targets).sum().item()
        total_count += inputs.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    logger.info("Loading & preprocessing corpus...")
    raw_text, source_label = _load_corpus(args)
    token_sequences = preprocess_corpus(raw_text)
    if not token_sequences:
        raise RuntimeError("No usable tokens found in the corpus.")

    stats = corpus_stats(raw_text, token_sequences)
    logger.info(
        "Corpus: %d sentences, %d tokens, %d unique, %d chars  ->  %s",
        stats["sentences"],
        stats["tokens"],
        stats["unique_tokens"],
        stats["characters"],
        source_label,
    )

    logger.info("Building tokenizer (min_freq=%d, max_vocab=%d)...", args.min_freq, args.max_vocab)
    tokenizer = _build_tokenizer(token_sequences, args.min_freq, args.max_vocab)
    logger.info("Vocabulary size: %d", tokenizer.vocab_size)

    # Encode the full corpus as one long id stream and slice into windows.
    full_stream: List[int] = []
    for seq in token_sequences:
        full_stream.extend(tokenizer.encode(seq))
        full_stream.append(tokenizer.eos_idx)
    inputs, targets = build_sliding_windows(full_stream, args.sequence_length)
    logger.info(
        "Sliding windows: %d (sequence_length=%d)", len(inputs), args.sequence_length
    )
    if len(inputs) < 100:
        logger.warning(
            "Very few training samples (%d). Consider lowering --sequence-length "
            "or supplying a larger corpus.",
            len(inputs),
        )

    train_loader, val_loader, test_loader = make_dataloaders(
        inputs, targets, args.batch_size, args.val_split, args.test_split, seed=args.seed
    )

    config = LSTMConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        pad_idx=tokenizer.pad_idx,
    )

    model = LSTMPredictor(config).to(device)
    logger.info(
        "Model: vocab=%d embed=%d hidden=%d layers=%d dropout=%.2f seq=%d  ->  %s",
        config.vocab_size,
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout,
        config.sequence_length,
        device,
    )
    logger.info("Trainable parameters: %d", model.num_parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    # AdamW with weight decay: decays all weights (including biases & norms)
    # is a known anti-pattern; we follow the standard advice and only decay
    # the parameters that aren't biases or layer norms.  This is a small
    # but real improvement over plain Adam.
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=1e-6,
        )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_perplexity": [],
        "learning_rate": [],
        "elapsed_seconds": 0.0,
        "corpus": source_label,
        "corpus_stats": stats,
        "args": vars(args),
    }

    best_val_loss = math.inf
    best_state = None
    best_epoch: int = 0
    epochs_without_improvement = 0
    start = time.time()

    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ppl = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_perplexity"].append(val_ppl)
        history["learning_rate"].append(current_lr)

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_ppl=%.2f | lr=%.2e",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_ppl,
            current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                logger.info(
                    "Early stopping: no val_loss improvement for %d epochs.",
                    epochs_without_improvement,
                )
                stopped_early = True
                break

    history["elapsed_seconds"] = round(time.time() - start, 2)
    history["best_epoch"] = best_epoch
    history["stopped_early"] = stopped_early

    # Restore best checkpoint before saving.
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final test evaluation -------------------------------------------
    test_loss, test_acc, test_ppl = evaluate(model, test_loader, criterion, device)
    history["test_loss"] = test_loss
    history["test_accuracy"] = test_acc
    history["test_perplexity"] = test_ppl
    logger.info(
        "Test set: loss=%.4f acc=%.4f perplexity=%.2f", test_loss, test_acc, test_ppl
    )

    # --- Persist artifacts -----------------------------------------------
    model_path = MODEL_DIR / args.save_model_name
    tokenizer_path = MODEL_DIR / args.save_tokenizer_name
    config_path = MODEL_DIR / args.save_config_name
    history_path = MODEL_DIR / "training_history.json"

    torch.save(model.state_dict(), model_path)
    tokenizer.save(str(tokenizer_path))
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2)
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    logger.info("Saved weights      -> %s", model_path)
    logger.info("Saved tokenizer    -> %s", tokenizer_path)
    logger.info("Saved config       -> %s", config_path)
    logger.info("Saved history      -> %s", history_path)


if __name__ == "__main__":
    main()
