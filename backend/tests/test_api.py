"""End-to-end tests for the FastAPI endpoints.

The API is exercised with `TestClient`.  Model-dependent endpoints are
covered with a tiny randomly-initialized model so the test suite runs on
any machine in a few seconds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.config import CONFIG_PATH, MODEL_PATH, TOKENIZER_PATH
from app.models.lstm_model import LSTMConfig, LSTMPredictor
from app.services.model_loader import model_manager
from app.utils.text_processing import Tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_artifacts(tmp_path_factory) -> dict:
    """Create a tiny trained model + tokenizer on disk and patch config paths.

    The backend reads paths from `app.config`, so we overwrite those module
    attributes to point at the temporary files for the duration of the
    tests.  Module reload would be heavier than this approach.
    """
    tmp = tmp_path_factory.mktemp("artifacts")

    vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "the", "future", "of", "is"]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    tokenizer = Tokenizer()
    tokenizer.word2idx = word2idx
    tokenizer.idx2word = idx2word
    tokenizer._frozen = True
    tokenizer.save(str(tmp / "tokenizer.pkl"))

    config = LSTMConfig(
        vocab_size=len(vocab),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        sequence_length=3,
        pad_idx=0,
    )
    model = LSTMPredictor(config)
    torch.save(model.state_dict(), str(tmp / "lstm_model.pth"))
    with open(tmp / "config.json", "w") as fh:
        json.dump(config.to_dict(), fh)

    # Patch module-level constants *before* the app starts.
    app_config = __import__("app.config", fromlist=["MODEL_PATH", "TOKENIZER_PATH", "CONFIG_PATH"])
    original = {
        "MODEL_PATH": app_config.MODEL_PATH,
        "TOKENIZER_PATH": app_config.TOKENIZER_PATH,
        "CONFIG_PATH": app_config.CONFIG_PATH,
    }
    app_config.MODEL_PATH = tmp / "lstm_model.pth"
    app_config.TOKENIZER_PATH = tmp / "tokenizer.pkl"
    app_config.CONFIG_PATH = tmp / "config.json"

    # Force the manager to reload.
    model_manager._loaded = None

    yield {"tmp": tmp}

    # Restore.
    for key, value in original.items():
        setattr(app_config, key, value)
    model_manager._loaded = None


@pytest.fixture
def client(tiny_artifacts) -> TestClient:
    from app.main import app  # imported lazily so patches apply

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"]


def test_health_reports_model_loaded(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["tokenizer_loaded"] is True


def test_model_info(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["architecture"] == "LSTM"
    assert body["framework"] == "PyTorch"
    assert body["vocab_size"] > 0
    assert body["device"] in {"cpu", "cuda"}


def test_predict_rejects_empty_input(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "   ", "top_k": 5})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower() or "required" in r.json()["detail"].lower()


def test_predict_rejects_invalid_top_k(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "the future of", "top_k": 0})
    assert r.status_code == 400


def test_predict_returns_predictions(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "the future of", "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_text"]
    assert isinstance(body["predictions"], list)
    assert 1 <= len(body["predictions"]) <= 3
    for pred in body["predictions"]:
        assert "word" in pred
        assert "probability" in pred
        assert 0.0 <= pred["probability"] <= 1.0
    # Sorted descending by probability.
    probs = [p["probability"] for p in body["predictions"]]
    assert probs == sorted(probs, reverse=True)


def test_generate_rejects_empty_input(client: TestClient) -> None:
    r = client.post("/generate", json={"text": "", "num_words": 5})
    assert r.status_code == 400


def test_generate_returns_words(client: TestClient) -> None:
    r = client.post("/generate", json={"text": "the future of", "num_words": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_text"]
    assert isinstance(body["generated_words"], list)
    assert len(body["generated_words"]) == 3
    assert body["generated_text"].startswith(body["input_text"])
