"""Unit and integration tests for test evaluation pipeline."""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.model.lstm import LSTMLanguageModel
from backend.training.dataset import CausalLanguageModelDataset
from backend.training.serialize import load_memmap_tokens
from backend.training.test_evaluate import (
    EPOCH_46_VAL_BASELINE,
    generate_reports,
    run_test_evaluation,
    validate_prerequisites,
)


def test_validate_prerequisites():
    """Verify that validate_prerequisites succeeds on real project files."""
    ckpt_path = "backend/model_artifacts/checkpoints/best_val.pt"
    data_dir = "backend/datasets/processed"
    vocab_path = "backend/model_artifacts/tokenizer/vocab.json"

    checkpoint, metadata, tokenizer, test_bin, test_offsets = validate_prerequisites(
        checkpoint_path=ckpt_path,
        data_dir=data_dir,
        vocab_path=vocab_path,
    )

    assert checkpoint["epoch"] == 46
    assert "model_state_dict" in checkpoint
    assert checkpoint["config"]["vocab_size"] == 30000
    assert len(tokenizer) == 30000
    assert test_bin.exists()
    assert test_offsets.exists()
    assert metadata["statistics"]["test"]["tokens"] == 7971938


def test_model_parameter_count_and_state_dict():
    """Verify model parameter count and successful state dict loading."""
    ckpt_path = "backend/model_artifacts/checkpoints/best_val.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]

    model = LSTMLanguageModel(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_idx=cfg["pad_idx"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 26748208

    # Ensure weights load without error
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


def test_run_test_evaluation_smoke():
    """Verify that run_test_evaluation calculates valid metrics on CPU for a few batches."""
    device = torch.device("cpu")
    ckpt_path = "backend/model_artifacts/checkpoints/best_val.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]

    model = LSTMLanguageModel(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_idx=cfg["pad_idx"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_tokens = load_memmap_tokens("backend/datasets/processed/test.bin")
    dataset = CausalLanguageModelDataset(
        token_data=test_tokens,
        seq_len=cfg["context_length"],
        stride=cfg["context_length"],
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["pad_idx"])

    metrics = run_test_evaluation(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        pad_idx=cfg["pad_idx"],
        max_batches=2,
    )

    assert metrics["test_loss"] > 0
    assert metrics["test_perplexity"] > 0
    assert 0.0 <= metrics["test_top1_acc"] <= metrics["test_top3_acc"] <= metrics["test_top5_acc"] <= 1.0
    assert metrics["total_valid_tokens"] == 2 * 32 * cfg["context_length"]


def test_report_generation(tmp_path: Path):
    """Verify JSON and Markdown report generation."""
    dummy_metrics = {
        "test_loss": 3.9174,
        "test_perplexity": 50.27,
        "test_top1_acc": 0.3244,
        "test_top3_acc": 0.4750,
        "test_top5_acc": 0.5397,
        "total_valid_tokens": 7971915,
        "total_raw_tokens": 7971915,
        "duration_seconds": 120.0,
        "throughput_tokens_per_sec": 66432.0,
        "total_batches": 890,
    }

    dummy_cfg = {
        "vocab_size": 30000,
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.3,
        "context_length": 35,
        "pad_idx": 0,
    }

    json_path, md_path = generate_reports(
        test_metrics=dummy_metrics,
        val_baseline=EPOCH_46_VAL_BASELINE,
        model_cfg=dummy_cfg,
        dataset_metadata={"statistics": {"test": {"documents": 2169, "tokens": 7971938}}},
        device_info={"device_type": "cpu", "device_name": "CPU"},
        checkpoint_info={"checkpoint_path": "best_val.pt", "epoch": 46},
        output_dir=tmp_path,
    )

    assert json_path.exists()
    assert md_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["test_metrics"]["test_loss"] == 3.9174
    assert data["generalization_gap"]["loss_diff"] < 0


def test_compute_top_k_accuracy_empty_and_pad_mask():
    """Verify compute_top_k_accuracy handles empty inputs and pad_idx masking correctly."""
    from backend.training.evaluate import compute_top_k_accuracy

    # Empty tensor inputs should return 0.0 without NaN
    empty_logits = torch.empty((0, 100))
    empty_targets = torch.empty((0,), dtype=torch.long)
    assert compute_top_k_accuracy(empty_logits, empty_targets, k=1) == 0.0

    # Test pad_idx masking: targets has 2 padding tokens (0) and 2 real tokens
    logits = torch.tensor([
        [10.0, 0.0, 0.0],  # predicts 0 (pad)
        [10.0, 0.0, 0.0],  # predicts 0 (pad)
        [0.0, 10.0, 0.0],  # predicts 1 (correct)
        [0.0, 0.0, 10.0],  # predicts 2 (correct)
    ])
    targets = torch.tensor([0, 0, 1, 2])
    # With pad_idx=0, only indices 2 and 3 are valid targets, both correct -> accuracy 1.0 (100%)
    acc_masked = compute_top_k_accuracy(logits, targets, k=1, pad_idx=0)
    assert acc_masked == 1.0

