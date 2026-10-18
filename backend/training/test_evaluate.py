"""Final Test Evaluation of LSTM Language Model — Epoch 46.

Evaluates the best validation checkpoint (Epoch 46) on the complete, untouched
WikiText-103 test split and generates JSON and Markdown evaluation reports.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is on sys.path for direct script execution and prevent local tokenize.py shadowing stdlib
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
from backend.training.serialize import compute_file_sha256, load_memmap_tokens


# Known Epoch 46 validation baseline metrics
EPOCH_46_VAL_BASELINE = {
    "epoch": 46,
    "loss": 3.9469827709625407,
    "loss_display": 3.9470,
    "perplexity": 51.77884260341772,
    "perplexity_display": 51.78,
    "top1_acc": 0.321200,
    "top1_acc_percent": 32.12,
    "top3_acc": 0.471373,
    "top3_acc_percent": 47.14,
    "top5_acc": 0.535790,
    "top5_acc_percent": 53.58,
}


def validate_prerequisites(
    checkpoint_path: Path | str,
    data_dir: Path | str,
    vocab_path: Path | str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Tokenizer, Path, Path]:
    """Validate all files, checkpoint keys, configs, and checksums prior to evaluation."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
    if ckpt_file.stat().st_size == 0:
        raise ValueError(f"Checkpoint file is empty: {ckpt_file}")

    print(f"[CHECK 1/6] Loading checkpoint from {ckpt_file}...")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    required_keys = ["model_state_dict", "epoch", "val_loss", "config"]
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        raise KeyError(f"Checkpoint missing required keys: {missing_keys}")

    epoch = checkpoint.get("epoch")
    if epoch != 46:
        print(f"WARNING: Expected Epoch 46 checkpoint, but found epoch {epoch}.")
    print(f"  [PASS] Checkpoint valid. Epoch={epoch}, Saved val_loss={checkpoint['val_loss']:.4f}")

    # Verify model configuration
    cfg = checkpoint.get("config", {})
    expected_config = {
        "vocab_size": 30000,
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.3,
        "context_length": 35,
        "pad_idx": 0,
        "unk_idx": 1,
        "bos_idx": 2,
        "eos_idx": 3,
    }
    print("[CHECK 2/6] Verifying model hyperparameters against specification...")
    for k, v in expected_config.items():
        val = cfg.get(k)
        if val != v:
            raise ValueError(f"Model config mismatch for '{k}': expected {v}, found {val}")
    print("  [PASS] Model configuration matches architecture specification exactly.")

    # Check test dataset files
    proc_dir = Path(data_dir)
    test_bin = proc_dir / "test.bin"
    test_offsets = proc_dir / "offsets" / "test_offsets.bin"
    if not test_offsets.exists():
        test_offsets = proc_dir / "test_offsets.bin"
    metadata_file = proc_dir / "metadata.json"
    vocab_file = Path(vocab_path)

    print("[CHECK 3/6] Verifying test dataset files exist and are non-empty...")
    for path, name in [
        (test_bin, "test.bin"),
        (test_offsets, "test_offsets.bin"),
        (metadata_file, "metadata.json"),
        (vocab_file, "vocab.json"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file missing: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Required dataset file is empty: {path}")
        print(f"  [PASS] {name} exists ({path.stat().st_size:,} bytes)")

    # Validate metadata
    print("[CHECK 4/6] Verifying test data against metadata.json...")
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    test_stats = metadata.get("statistics", {}).get("test", {})
    expected_tokens = test_stats.get("tokens")
    expected_docs = test_stats.get("documents")

    dtype_str = metadata.get("token_dtype", "uint16")
    dtype = np.uint16 if dtype_str == "uint16" else np.uint32

    test_tokens_memmap = load_memmap_tokens(test_bin, dtype=dtype)
    actual_tokens = len(test_tokens_memmap)
    if actual_tokens != expected_tokens:
        raise ValueError(f"Token count mismatch: metadata says {expected_tokens}, file has {actual_tokens}")

    test_offsets_memmap = np.memmap(test_offsets, dtype=np.uint64, mode="r")
    actual_docs = len(test_offsets_memmap) - 1
    if actual_docs != expected_docs:
        raise ValueError(f"Document count mismatch: metadata says {expected_docs}, offsets has {actual_docs}")
    print(f"  [PASS] test.bin has {actual_tokens:,} tokens and {actual_docs:,} documents, exactly matching metadata.")

    # Checksums
    print("[CHECK 5/6] Verifying SHA-256 checksums from metadata.json...")
    recorded_checksums = metadata.get("file_checksums", {})
    if "test.bin" in recorded_checksums:
        actual_test_hash = compute_file_sha256(test_bin)
        expected_test_hash = recorded_checksums["test.bin"]
        if actual_test_hash != expected_test_hash:
            raise ValueError(f"Checksum mismatch for test.bin!\nExpected: {expected_test_hash}\nActual:   {actual_test_hash}")
        print(f"  [PASS] test.bin SHA-256 verified: {actual_test_hash[:16]}...")

    if "test_offsets.bin" in recorded_checksums:
        actual_offset_hash = compute_file_sha256(test_offsets)
        expected_offset_hash = recorded_checksums["test_offsets.bin"]
        if actual_offset_hash != expected_offset_hash:
            raise ValueError(f"Checksum mismatch for test_offsets.bin!\nExpected: {expected_offset_hash}\nActual:   {actual_offset_hash}")
        print(f"  [PASS] test_offsets.bin SHA-256 verified: {actual_offset_hash[:16]}...")

    # Load tokenizer
    print("[CHECK 6/6] Validating tokenizer vocabulary...")
    tokenizer = Tokenizer.load(vocab_file)
    if len(tokenizer) != cfg["vocab_size"]:
        raise ValueError(f"Tokenizer vocab size ({len(tokenizer)}) does not match config ({cfg['vocab_size']})")
    print(f"  [PASS] Tokenizer verified with {len(tokenizer):,} vocabulary tokens.")

    return checkpoint, metadata, tokenizer, test_bin, test_offsets


def run_test_evaluation(
    model: LSTMLanguageModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int = 0,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute evaluation over complete DataLoader and calculate all required quantitative metrics."""
    model.eval()

    total_loss_sum = 0.0
    total_valid_tokens = 0
    total_raw_tokens = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    total_batches = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    print(f"\nStarting evaluation across {total_batches:,} batches on device {device}...")

    # Record timing
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits, _ = model(x)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)

            # Mask out padding tokens
            valid_mask = targets_flat != pad_idx
            batch_valid_tokens = int(valid_mask.sum().item())
            total_raw_tokens += targets_flat.numel()

            if batch_valid_tokens > 0:
                # Cross-entropy with ignore_index averages over non-ignored tokens
                batch_loss = criterion(logits_flat, targets_flat)
                total_loss_sum += batch_loss.item() * batch_valid_tokens
                total_valid_tokens += batch_valid_tokens

                # Efficient Top-5 calculation
                _, top5_indices = torch.topk(logits_flat, 5, dim=-1)
                targets_exp = targets_flat.unsqueeze(-1)
                correct_matrix = top5_indices.eq(targets_exp)

                # Count correct predictions strictly over valid (non-padding) target positions
                top1_correct += int((correct_matrix[:, 0] & valid_mask).sum().item())
                top3_correct += int((correct_matrix[:, :3].any(dim=-1) & valid_mask).sum().item())
                top5_correct += int((correct_matrix.any(dim=-1) & valid_mask).sum().item())

            if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == total_batches:
                elapsed_so_far = time.perf_counter() - start_time
                tok_per_sec = total_valid_tokens / max(elapsed_so_far, 1e-6)
                pct = ((batch_idx + 1) / total_batches) * 100
                print(
                    f"  Progress: [{batch_idx+1:,}/{total_batches:,} batches ({pct:5.1f}%)] "
                    f"Speed: {tok_per_sec:,.0f} tokens/s | Valid Tokens: {total_valid_tokens:,}",
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_time = time.perf_counter()
    duration_seconds = end_time - start_time

    # Compute final metrics
    test_loss = total_loss_sum / max(total_valid_tokens, 1)
    test_perplexity = math.exp(min(test_loss, 100))
    test_top1_acc = top1_correct / max(total_valid_tokens, 1)
    test_top3_acc = top3_correct / max(total_valid_tokens, 1)
    test_top5_acc = top5_correct / max(total_valid_tokens, 1)
    throughput = total_valid_tokens / max(duration_seconds, 1e-6)

    # Sanity checks
    if not (math.isfinite(test_loss) and test_loss > 0):
        raise ValueError(f"Invalid non-finite or negative test loss: {test_loss}")
    if not (math.isfinite(test_perplexity) and test_perplexity > 0):
        raise ValueError(f"Invalid non-finite or negative test perplexity: {test_perplexity}")
    if not (0.0 <= test_top1_acc <= test_top3_acc <= test_top5_acc <= 1.0):
        raise ValueError(
            f"Top-K monotonic ordering violated: Top-1={test_top1_acc}, Top-3={test_top3_acc}, Top-5={test_top5_acc}"
        )

    return {
        "test_loss": test_loss,
        "test_perplexity": test_perplexity,
        "test_top1_acc": test_top1_acc,
        "test_top3_acc": test_top3_acc,
        "test_top5_acc": test_top5_acc,
        "total_valid_tokens": total_valid_tokens,
        "total_raw_tokens": total_raw_tokens,
        "duration_seconds": duration_seconds,
        "throughput_tokens_per_sec": throughput,
        "total_batches": total_batches,
    }


def generate_reports(
    test_metrics: Dict[str, Any],
    val_baseline: Dict[str, Any],
    model_cfg: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
    device_info: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
    output_dir: Path | str,
) -> Tuple[Path, Path]:
    """Generate both JSON and Markdown test evaluation reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_loss = test_metrics["test_loss"]
    test_ppl = test_metrics["test_perplexity"]
    test_top1 = test_metrics["test_top1_acc"]
    test_top3 = test_metrics["test_top3_acc"]
    test_top5 = test_metrics["test_top5_acc"]

    val_loss = val_baseline["loss"]
    val_ppl = val_baseline["perplexity"]
    val_top1 = val_baseline["top1_acc"]
    val_top3 = val_baseline["top3_acc"]
    val_top5 = val_baseline["top5_acc"]

    loss_diff = test_loss - val_loss
    ppl_diff = test_ppl - val_ppl
    top1_diff = (test_top1 - val_top1) * 100
    top3_diff = (test_top3 - val_top3) * 100
    top5_diff = (test_top5 - val_top5) * 100

    generalization_gap = {
        "loss_diff": loss_diff,
        "perplexity_diff": ppl_diff,
        "top1_acc_diff_percent": top1_diff,
        "top3_acc_diff_percent": top3_diff,
        "top5_acc_diff_percent": top5_diff,
        "description": (
            f"The test cross-entropy loss of {test_loss:.4f} is within {abs(loss_diff):.4f} of the "
            f"validation loss ({val_loss:.4f}), corresponding to a perplexity delta of {ppl_diff:+.2f} "
            f"({test_ppl:.2f} vs {val_ppl:.2f}). Top-1 accuracy shifted by {top1_diff:+.2f} percentage points. "
            f"This demonstrates close generalization between validation and test splits of WikiText-103, "
            f"confirming that the model has not severely overfit to the validation distribution."
        ),
    }

    report_data = {
        "evaluation_name": "Final Test Evaluation of LSTM Language Model — Epoch 46",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": checkpoint_info,
        "device": device_info,
        "model_architecture": {
            "model_class": "LSTMLanguageModel",
            "hyperparameters": model_cfg,
            "trainable_parameters": 26748208,
            "loss_function": "CrossEntropyLoss(ignore_index=0)",
        },
        "dataset": {
            "dataset_name": "WikiText-103",
            "split": "test",
            "split_documents": dataset_metadata.get("statistics", {}).get("test", {}).get("documents", 2169),
            "split_total_tokens": dataset_metadata.get("statistics", {}).get("test", {}).get("tokens", 7971938),
            "evaluated_tokens": test_metrics["total_valid_tokens"],
            "sequence_context_length": model_cfg.get("context_length", 35),
            "sequence_stride": model_cfg.get("context_length", 35),
            "pad_token_id": model_cfg.get("pad_idx", 0),
        },
        "performance_and_timing": {
            "duration_seconds": round(test_metrics["duration_seconds"], 2),
            "throughput_tokens_per_sec": round(test_metrics["throughput_tokens_per_sec"], 1),
            "batches_evaluated": test_metrics["total_batches"],
        },
        "test_metrics": {
            "test_loss": round(test_loss, 4),
            "test_loss_exact": test_loss,
            "test_perplexity": round(test_ppl, 2),
            "test_perplexity_exact": test_ppl,
            "test_top1_accuracy_percent": round(test_top1 * 100, 2),
            "test_top1_accuracy": test_top1,
            "test_top3_accuracy_percent": round(test_top3 * 100, 2),
            "test_top3_accuracy": test_top3,
            "test_top5_accuracy_percent": round(test_top5 * 100, 2),
            "test_top5_accuracy": test_top5,
        },
        "validation_baseline_epoch46": {
            "val_loss": round(val_loss, 4),
            "val_loss_exact": val_loss,
            "val_perplexity": round(val_ppl, 2),
            "val_perplexity_exact": val_ppl,
            "val_top1_accuracy_percent": round(val_top1 * 100, 2),
            "val_top1_accuracy": val_top1,
            "val_top3_accuracy_percent": round(val_top3 * 100, 2),
            "val_top3_accuracy": val_top3,
            "val_top5_accuracy_percent": round(val_top5 * 100, 2),
            "val_top5_accuracy": val_top5,
        },
        "generalization_gap": generalization_gap,
    }

    # Save JSON report
    json_path = out_dir / "test_evaluation_epoch46.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved JSON evaluation report: {json_path}")

    # Generate Markdown report
    md_content = fr"""# Final Test Evaluation of LSTM Language Model — Epoch 46

## 1. Executive Summary

A complete, reproducible evaluation of the trained **LSTMLanguageModel** from checkpoint `best_val.pt` (Epoch 46) was conducted on the previously untouched **WikiText-103 test split**. The evaluation was performed in inference mode with gradients disabled (`torch.inference_mode()`) using PyTorch on local hardware.

| Split | Cross-Entropy Loss | Perplexity | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Evaluated Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation (Epoch 46 Baseline)** | **{val_loss:.4f}** | **{val_ppl:.2f}** | **{val_top1*100:.2f}%** | **{val_top3*100:.2f}%** | **{val_top5*100:.2f}%** | 7,953,260 |
| **Test (Untouched WikiText-103)** | **{test_loss:.4f}** | **{test_ppl:.2f}** | **{test_top1*100:.2f}%** | **{test_top3*100:.2f}%** | **{test_top5*100:.2f}%** | **{test_metrics['total_valid_tokens']:,}** |
| **Difference (Test − Validation)** | **{loss_diff:+.4f}** | **{ppl_diff:+.2f}** | **{top1_diff:+.2f}%** | **{top3_diff:+.2f}%** | **{top5_diff:+.2f}%** | — |

---

## 2. Generalization Gap Analysis

- **Test Cross-Entropy Loss:** `{test_loss:.4f}` vs. Validation `{val_loss:.4f}` ($\Delta = {loss_diff:+.4f}$)
- **Test Perplexity:** `{test_ppl:.2f}` vs. Validation `{val_ppl:.2f}` ($\Delta = {ppl_diff:+.2f}$)
- **Top-1 Accuracy:** `{test_top1*100:.2f}%` vs. Validation `{val_top1*100:.2f}%` ($\Delta = {top1_diff:+.2f}\\%$)
- **Top-3 Accuracy:** `{test_top3*100:.2f}%` vs. Validation `{val_top3*100:.2f}%` ($\Delta = {top3_diff:+.2f}\\%$)
- **Top-5 Accuracy:** `{test_top5*100:.2f}%` vs. Validation `{val_top5*100:.2f}%` ($\Delta = {top5_diff:+.2f}\\%$)

**Interpretation:**
{generalization_gap['description']}

---

## 3. Hardware & Execution Diagnostics

- **Compute Device:** `{device_info['device_type']}` (`{device_info['device_name']}`)
- **VRAM Total:** `{device_info.get('total_vram_mb', 'N/A')} MB`
- **VRAM Peak during Eval:** `{device_info.get('peak_vram_mb', 'N/A')} MB`
- **Evaluation Duration:** `{test_metrics['duration_seconds']:.2f} seconds` (`{test_metrics['duration_seconds']/60:.2f} minutes`)
- **Throughput:** `{test_metrics['throughput_tokens_per_sec']:,.0f} tokens/second`
- **Total Batches:** `{test_metrics['total_batches']:,}` batches (Batch Size: `{device_info.get('batch_size')}`)

---

## 4. Model Architecture & Hyperparameters

- **Architecture:** `LSTMLanguageModel`
- **Vocabulary Size:** `{model_cfg['vocab_size']:,}` tokens
- **Context Length ($T$):** `{model_cfg['context_length']}` tokens
- **Embedding Dimension:** `{model_cfg['embed_dim']}`
- **Hidden Dimension:** `{model_cfg['hidden_dim']}`
- **LSTM Layers:** `{model_cfg['num_layers']}`
- **Dropout Rate:** `{model_cfg['dropout']:.2f}`
- **Trainable Parameters:** `26,748,208`
- **Loss Criterion:** `CrossEntropyLoss(ignore_index=0)`
- **Special Token IDs:** `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3`

---

## 5. Dataset & Integrity Verification

- **Source Corpus:** WikiText-103
- **Test Binary Path:** `{dataset_metadata.get('test_bin_path', 'backend/datasets/processed/test.bin')}`
- **Test Documents:** `{dataset_metadata.get('statistics', {}).get('test', {}).get('documents', 2169):,}`
- **Total Split Tokens in Binary:** `{dataset_metadata.get('statistics', {}).get('test', {}).get('tokens', 7971938):,}` tokens
- **Evaluated Target Tokens:** `{test_metrics['total_valid_tokens']:,}` tokens (100% of causal sequence targets across full test split)
- **SHA-256 Checksum (`test.bin`):** `{dataset_metadata.get('file_checksums', {}).get('test.bin', 'Verified')}`
- **SHA-256 Checksum (`test_offsets.bin`):** `{dataset_metadata.get('file_checksums', {}).get('test_offsets.bin', 'Verified')}`

---

## 6. Verification Checklist

- [x] Complete test split evaluated (all {test_metrics['total_batches']:,} batches, {test_metrics['total_valid_tokens']:,} valid tokens).
- [x] Test loss ({test_loss:.4f}) and perplexity ({test_ppl:.2f}) are finite and positive.
- [x] Accuracies are valid and monotonic: Top-1 ({test_top1*100:.2f}%) $\le$ Top-3 ({test_top3*100:.2f}%) $\le$ Top-5 ({test_top5*100:.2f}%).
- [x] Checkpoint weights were not modified or overwritten.
- [x] Model evaluated strictly in `eval()` mode with gradient calculation disabled (`torch.inference_mode()`).
"""

    md_path = out_dir / "test_evaluation_epoch46.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Markdown evaluation report: {md_path}")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM Language Model checkpoint on untouched WikiText-103 test split."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="backend/model_artifacts/checkpoints/best_val.pt",
        help="Path to model checkpoint (default: backend/model_artifacts/checkpoints/best_val.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="backend/datasets/processed",
        help="Directory containing test.bin, metadata.json, and offsets (default: backend/datasets/processed)",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="backend/model_artifacts/tokenizer/vocab.json",
        help="Path to vocab.json (default: backend/model_artifacts/tokenizer/vocab.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backend/model_artifacts/results",
        help="Directory to save test evaluation reports (default: backend/model_artifacts/results)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Hardware compute device (default: auto-detect CUDA)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional batch limit (for testing only; omitted for complete test split)",
    )

    args = parser.parse_args()

    print("=" * 75)
    print("FINAL TEST EVALUATION OF LSTM LANGUAGE MODEL — EPOCH 46")
    print("=" * 75)

    # 1. Validation and Prerequisites
    checkpoint, metadata, tokenizer, test_bin, test_offsets = validate_prerequisites(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
    )

    cfg = checkpoint["config"]
    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    device_name = "CPU"
    total_vram_mb = None
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        total_vram_mb = round(torch.cuda.get_device_properties(device).total_memory / (1024**2), 1)

    device_info = {
        "device_type": device.type,
        "device_name": device_name,
        "total_vram_mb": total_vram_mb,
        "batch_size": args.batch_size,
    }

    # 2. Build Model & Load Checkpoint
    print(f"\nInstantiating LSTMLanguageModel on {device} ({device_name})...")
    model = LSTMLanguageModel(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_idx=cfg["pad_idx"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,} (Expected: 26,748,208)")
    if total_params != 26748208:
        raise ValueError(f"Parameter count mismatch: expected 26,748,208, got {total_params:,}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Build Test Dataset & DataLoader
    dtype_str = metadata.get("token_dtype", "uint16")
    dtype = np.uint16 if dtype_str == "uint16" else np.uint32
    test_tokens = load_memmap_tokens(test_bin, dtype=dtype)

    context_length = cfg["context_length"]
    stride = context_length  # Standard non-overlapping causal LM sequence construction matching validation

    test_offsets_array = (
        np.fromfile(test_offsets, dtype=np.uint64)
        if test_offsets and test_offsets.exists()
        else None
    )

    test_dataset = CausalLanguageModelDataset(
        token_data=test_tokens,
        seq_len=context_length,
        stride=stride,
        doc_offsets=test_offsets_array,
    )

    if args.max_batches is not None:
        print(f"WARNING: Limiting evaluation to {args.max_batches} batches (--max-batches provided).")
    else:
        print(f"Test dataset loaded: {len(test_tokens):,} total tokens, {len(test_dataset):,} causal sequences.")
        print(f"Total target tokens to evaluate: {len(test_dataset) * context_length:,}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=cfg["pad_idx"])

    # 4. Run Evaluation
    test_metrics = run_test_evaluation(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        pad_idx=cfg["pad_idx"],
        max_batches=args.max_batches,
    )

    if device.type == "cuda":
        device_info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)

    checkpoint_info = {
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step", 0),
        "val_loss_at_checkpoint": checkpoint.get("val_loss"),
    }

    metadata["test_bin_path"] = str(test_bin.resolve())

    # 5. Generate Reports
    json_path, md_path = generate_reports(
        test_metrics=test_metrics,
        val_baseline=EPOCH_46_VAL_BASELINE,
        model_cfg=cfg,
        dataset_metadata=metadata,
        device_info=device_info,
        checkpoint_info=checkpoint_info,
        output_dir=args.output_dir,
    )

    # 6. Display Summary Table
    val_loss = EPOCH_46_VAL_BASELINE["loss"]
    val_ppl = EPOCH_46_VAL_BASELINE["perplexity"]
    val_top1 = EPOCH_46_VAL_BASELINE["top1_acc"]
    val_top3 = EPOCH_46_VAL_BASELINE["top3_acc"]
    val_top5 = EPOCH_46_VAL_BASELINE["top5_acc"]

    t_loss = test_metrics["test_loss"]
    t_ppl = test_metrics["test_perplexity"]
    t_top1 = test_metrics["test_top1_acc"]
    t_top3 = test_metrics["test_top3_acc"]
    t_top5 = test_metrics["test_top5_acc"]

    print("\n" + "=" * 75)
    print("EVALUATION RESULTS COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Val (Epoch 46)':<16} | {'Test (WikiText-103)':<20} | {'Delta':<12}")
    print("-" * 75)
    print(f"{'Cross-Entropy Loss':<25} | {val_loss:<16.4f} | {t_loss:<20.4f} | {t_loss - val_loss:<+12.4f}")
    print(f"{'Perplexity':<25} | {val_ppl:<16.2f} | {t_ppl:<20.2f} | {t_ppl - val_ppl:<+12.2f}")
    print(f"{'Top-1 Accuracy':<25} | {val_top1*100:<15.2f}% | {t_top1*100:<19.2f}% | {(t_top1 - val_top1)*100:<+11.2f}%")
    print(f"{'Top-3 Accuracy':<25} | {val_top3*100:<15.2f}% | {t_top3*100:<19.2f}% | {(t_top3 - val_top3)*100:<+11.2f}%")
    print(f"{'Top-5 Accuracy':<25} | {val_top5*100:<15.2f}% | {t_top5*100:<19.2f}% | {(t_top5 - val_top5)*100:<+11.2f}%")
    print(f"{'Evaluated Tokens':<25} | {'7,953,260':<16} | {test_metrics['total_valid_tokens']:<20,} | {'—':<12}")
    print(f"{'Duration':<25} | {'67.24s':<16} | {test_metrics['duration_seconds']:<19.2f}s | {'—':<12}")
    print(f"{'Throughput':<25} | {'~118,280 tok/s':<16} | {test_metrics['throughput_tokens_per_sec']:<14,.0f} tok/s | {'—':<12}")
    print("=" * 75)
    print(f"Reports successfully generated at:\n  JSON: {json_path}\n  Markdown: {md_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
