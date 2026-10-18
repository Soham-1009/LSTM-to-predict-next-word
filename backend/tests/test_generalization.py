"""Tests for Multi-Domain Generalization, Evaluation Reproducibility, and Checkpoint Compatibility."""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from backend.app.main import app, service
from backend.model.lstm import LSTMLanguageModel
from backend.model.tokenizer import Tokenizer
from backend.training.evaluate import compute_top_k_accuracy


@pytest.fixture(scope="module")
def client():
    """TestClient fixture for FastAPI application."""
    return TestClient(app)


def test_generalization_benchmark_artifacts_exist_and_valid():
    """Verify that empirical benchmark artifacts exist and conform to schemas."""
    results_dir = Path("backend/model_artifacts/results")
    baseline_path = results_dir / "generalization_benchmark_baseline.json"
    adapted_path = results_dir / "generalization_benchmark_adapted.json"
    comparison_path = results_dir / "generalization_comparison.json"

    assert baseline_path.exists(), "baseline benchmark JSON must exist"
    assert adapted_path.exists(), "adapted benchmark JSON must exist"
    assert comparison_path.exists(), "comparison JSON must exist"

    with open(comparison_path, "r", encoding="utf-8") as f:
        comp = json.load(f)

    # Validate overall summary
    assert "overall_summary" in comp
    base_ov = comp["overall_summary"]["baseline"]
    adapt_ov = comp["overall_summary"]["adapted"]

    # Verify metric validity ranges
    for ov in [base_ov, adapt_ov]:
        assert ov["loss"] > 0.0
        assert ov["perplexity"] >= 1.0
        assert 0.0 <= ov["top1_acc"] <= ov["top5_acc"] <= 100.0
        assert 0.0 <= ov["oov_rate_pct"] <= 100.0

    # Verify measurable improvement on adapted model
    deltas = comp["overall_summary"]["deltas"]
    assert deltas["loss_delta"] < 0.0, "Adapted model must achieve lower multi-domain cross-entropy loss"
    assert deltas["ppl_delta"] < 0.0, "Adapted model must achieve lower multi-domain perplexity"
    assert deltas["top1_delta"] > 0.0, "Adapted model must achieve higher Top-1 accuracy"

    # Verify no catastrophic forgetting on in-domain WikiText-103
    retention = comp["in_domain_retention"]
    assert not retention["deltas"]["catastrophic_forgetting"], "Model must not suffer catastrophic forgetting"
    assert abs(retention["deltas"]["loss_delta"]) < 0.15, "WikiText retention loss shift must be under 0.15"

    # Validate macro-average and token-weighted summaries
    assert "macro_average_summary" in comp
    macro = comp["macro_average_summary"]
    assert macro["baseline"]["loss"] > 0.0
    assert macro["adapted"]["loss"] > 0.0
    assert macro["deltas"]["loss_delta"] < 0.0
    assert macro["deltas"]["ppl_delta"] < 0.0
    assert macro["deltas"]["top1_delta"] > 0.0

    assert "token_weighted_summary" in comp
    tw = comp["token_weighted_summary"]
    assert tw["baseline"]["evaluated_tokens"] == 21210
    assert tw["adapted"]["evaluated_tokens"] == 21210


def test_per_domain_coverage_and_stratification():
    """Verify that all 6 functional domains are evaluated with zero leakage."""
    comparison_path = Path("backend/model_artifacts/results/generalization_comparison.json")
    with open(comparison_path, "r", encoding="utf-8") as f:
        comp = json.load(f)

    expected_domains = {
        "conversation",
        "literature",
        "news",
        "scientific",
        "technical_docs",
        "wikitext103_replay",
    }
    actual_domains = {d["domain"] for d in comp["domain_breakdown"]}
    assert actual_domains == expected_domains, f"Missing domains: {expected_domains - actual_domains}"

    # Verify each domain has positive token count and valid metrics
    for d in comp["domain_breakdown"]:
        assert d["tokens"] > 0
        assert d["oov_rate_pct"] > 0.0
        assert d["vocab_coverage_pct"] == round(100.0 - d["oov_rate_pct"], 2)
        assert d["baseline"]["perplexity"] > 0.0
        assert d["adapted"]["perplexity"] > 0.0


def test_checkpoint_compatibility_and_vocabulary_preservation():
    """Verify that both baseline and adapted checkpoints load with exact same architecture and vocab."""
    ckpt_dir = Path("backend/model_artifacts/checkpoints")
    adapted_dir = Path("experiments/multidomain_v1/checkpoints")

    base_ckpt = torch.load(ckpt_dir / "best_val.pt", map_location="cpu", weights_only=True)
    adapt_ckpt = torch.load(adapted_dir / "multidomain_adapted.pt", map_location="cpu", weights_only=True)

    # Check configuration identity
    base_cfg = base_ckpt["config"]
    adapt_cfg = adapt_ckpt["config"]

    assert base_cfg["vocab_size"] == adapt_cfg["vocab_size"] == 30000
    assert base_cfg["embed_dim"] == adapt_cfg["embed_dim"] == 256
    assert base_cfg["hidden_dim"] == adapt_cfg["hidden_dim"] == 512
    assert base_cfg["num_layers"] == adapt_cfg["num_layers"] == 2
    assert base_cfg["pad_idx"] == adapt_cfg["pad_idx"] == 0

    # Ensure weights load into the exact same PyTorch model
    model = LSTMLanguageModel(
        vocab_size=30000,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        pad_idx=0,
    )

    # Test baseline forward pass
    model.load_state_dict(base_ckpt["model_state_dict"])
    dummy_input = torch.randint(1, 30000, (2, 10))
    logits_base, _ = model(dummy_input)
    assert logits_base.shape == (2, 10, 30000)

    # Test adapted forward pass
    model.load_state_dict(adapt_ckpt["model_state_dict"])
    logits_adapt, _ = model(dummy_input)
    assert logits_adapt.shape == (2, 10, 30000)


def test_fastapi_evaluation_endpoints(client):
    """Verify that FastAPI serves generalization and baseline evaluation reports."""
    # Test GET /evaluation/generalization
    res_gen = client.get("/evaluation/generalization")
    assert res_gen.status_code == 200
    gen_data = res_gen.json()
    assert "overall_summary" in gen_data
    assert "domain_breakdown" in gen_data
    assert len(gen_data["domain_breakdown"]) == 6
    assert gen_data["leakage_audit_summary"]["status"] == "AUDIT_PASSED"

    # Test GET /evaluation/test-baseline
    res_base = client.get("/evaluation/test-baseline")
    assert res_base.status_code == 200
    base_data = res_base.json()
    assert base_data["checkpoint"]["epoch"] == 46
    assert base_data["test_metrics"]["test_perplexity"] < 52.0


def test_fastapi_dynamic_checkpoint_switching(client):
    """Verify switching between baseline and adapted models via API and testing both predict and generate."""
    # Ensure starting at baseline
    res_switch_base = client.post("/model/select-checkpoint", json={"checkpoint_name": "baseline"})
    assert res_switch_base.status_code == 200
    assert res_switch_base.json()["active_checkpoint"] == "baseline"

    # Test predict with baseline
    res_pred_base = client.post("/predict", json={"text": "In computer science, a function", "top_k": 5})
    assert res_pred_base.status_code == 200
    assert len(res_pred_base.json()["predictions"]) == 5

    # Test generate with baseline
    res_gen_base = client.post("/generate", json={"prompt": "In computer science", "max_new_tokens": 10, "greedy": True})
    assert res_gen_base.status_code == 200
    assert len(res_gen_base.json()["generated_text"]) > len("In computer science")

    # Switch to adapted
    res_switch_adapt = client.post("/model/select-checkpoint", json={"checkpoint_name": "adapted"})
    assert res_switch_adapt.status_code == 200
    assert res_switch_adapt.json()["active_checkpoint"] == "adapted"

    # Verify model info reflects change
    info = client.get("/model-info").json()
    assert info["active_checkpoint"] == "adapted"

    # Test predict with adapted model
    res_pred_adapt = client.post("/predict", json={"text": "In computer science, a function", "top_k": 5})
    assert res_pred_adapt.status_code == 200
    preds = res_pred_adapt.json()["predictions"]
    assert len(preds) == 5
    assert sum(p["probability"] for p in preds) <= 1.01

    # Test generate with adapted model
    res_gen_adapt = client.post("/generate", json={"prompt": "In computer science", "max_new_tokens": 10, "greedy": True})
    assert res_gen_adapt.status_code == 200
    assert len(res_gen_adapt.json()["generated_text"]) > len("In computer science")

    # Switch back to baseline
    res_switch_back = client.post("/model/select-checkpoint", json={"checkpoint_name": "baseline"})
    assert res_switch_back.status_code == 200
    assert res_switch_back.json()["active_checkpoint"] == "baseline"

    # Test invalid checkpoint rejection
    res_invalid = client.post("/model/select-checkpoint", json={"checkpoint_name": "non_existent"})
    assert res_invalid.status_code in [400, 404]
