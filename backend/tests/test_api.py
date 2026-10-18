"""Unit and integration tests for backend API, tokenizer, and PyTorch LSTM model."""

import pytest
import torch
from fastapi.testclient import TestClient

try:
    from backend.app.main import app
    from backend.model.lstm import LSTMLanguageModel, sample_next_token
    from backend.model.tokenizer import Tokenizer
    from backend.model.config import ModelConfig
except ImportError:
    from app.main import app
    from model.lstm import LSTMLanguageModel, sample_next_token
    from model.tokenizer import Tokenizer
    from model.config import ModelConfig


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "version" in data


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "LSTM Causal Language Model"
    assert data["vocab_size"] > 0
    assert data["num_layers"] == 2
    assert data["hidden_dim"] == 512
    assert data["embed_dim"] == 256


def test_predict_endpoint_real_inference(client):
    response = client.post("/predict", json={"text": "Machine learning", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["input_text"] == "Machine learning"
    assert len(data["predictions"]) == 3
    # Check that predictions are valid tokens with probability between 0 and 1
    for pred in data["predictions"]:
        assert isinstance(pred["token"], str)
        assert 0.0 <= pred["probability"] <= 1.0


def test_predict_empty_text_validation(client):
    response = client.post("/predict", json={"text": "", "top_k": 3})
    assert response.status_code == 400 or response.status_code == 422


def test_generate_endpoint_real_inference(client):
    response = client.post(
        "/generate",
        json={"prompt": "deep neural", "max_new_tokens": 5, "temperature": 0.8},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "deep neural"
    assert len(data["generated_text"]) >= len("deep neural")
    assert data["tokens_generated"] >= 0


def test_tokenizer_encoding_and_decoding():
    tokenizer = Tokenizer()
    corpus = ["hello world this is an lstm language model"]
    # Build vocabulary from corpus
    freqs = {"hello": 5, "world": 5, "this": 4, "is": 4, "an": 3, "lstm": 3, "language": 2, "model": 2}
    tokenizer.build_vocab(freqs)

    sample_text = "hello world this is an lstm language model"
    token_ids = tokenizer.encode(sample_text, add_bos=True, add_eos=True)
    assert len(token_ids) > 2
    assert token_ids[0] == tokenizer.bos_id
    assert token_ids[-1] == tokenizer.eos_id

    # Decoding with skip_special_tokens=True should strip BOS and EOS
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert isinstance(decoded, str)
    assert "hello" in decoded
    assert "model" in decoded


def test_lstm_forward_pass():
    vocab_size = 1000
    embed_dim = 64
    hidden_dim = 128
    num_layers = 2
    batch_size = 4
    seq_len = 16

    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
    )
    model.eval()

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, hidden = model(dummy_input)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert hidden[0].shape == (num_layers, batch_size, hidden_dim)
    assert hidden[1].shape == (num_layers, batch_size, hidden_dim)


def test_sampling_methods():
    vocab_size = 50
    dummy_logits = torch.randn(vocab_size)

    # Greedy
    greedy_token = sample_next_token(dummy_logits, greedy=True)
    assert 0 <= greedy_token < vocab_size
    assert greedy_token == int(torch.argmax(dummy_logits).item())

    # Top-K
    top_k_token = sample_next_token(dummy_logits, temperature=0.8, top_k=5)
    assert 0 <= top_k_token < vocab_size

    # Top-P
    top_p_token = sample_next_token(dummy_logits, temperature=0.8, top_p=0.9)
    assert 0 <= top_p_token < vocab_size


def test_predict_short_input(client):
    """Test very short inputs such as a single word or single character."""
    for short_text in ["a", "I", "The"]:
        response = client.post("/predict", json={"text": short_text, "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3
        for pred in data["predictions"]:
            assert isinstance(pred["token"], str)
            assert 0.0 <= pred["probability"] <= 1.0


def test_predict_long_input_exceeding_context(client):
    """Test input text exceeding the 35-token context length."""
    long_prompt = "The quick brown fox jumps over the lazy dog and runs across the wide open fields " * 5
    response = client.post("/predict", json={"text": long_prompt, "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 5
    for pred in data["predictions"]:
        assert 0.0 <= pred["probability"] <= 1.0


def test_predict_unknown_words(client):
    """Test input consisting entirely of out-of-vocabulary words."""
    oov_text = "xyznonexistentwordqwerty zzzblorpkablooey"
    response = client.post("/predict", json={"text": oov_text, "top_k": 4})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 4


def test_predict_top_k_variations_and_ordering(client):
    """Test different Top-K values (1, 5, 20) and verify descending probability order."""
    for k in [1, 5, 20]:
        response = client.post("/predict", json={"text": "Artificial intelligence", "top_k": k})
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == k
        probs = [p["probability"] for p in data["predictions"]]
        # Probabilities should be descending or non-increasing
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]


def test_predict_suppresses_structural_tokens(client):
    """Ensure structural tokens (<PAD>, <UNK>, <BOS>, <EOS>) are suppressed in autocomplete."""
    response = client.post("/predict", json={"text": "The model", "top_k": 20})
    assert response.status_code == 200
    data = response.json()
    tokens = [p["token"] for p in data["predictions"]]
    structural_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
    for tok in tokens:
        assert tok not in structural_tokens


def test_predict_invalid_parameters(client):
    """Test parameter bounds validation for /predict."""
    # Empty / whitespace
    res_empty = client.post("/predict", json={"text": "   ", "top_k": 5})
    assert res_empty.status_code in [400, 422]

    # top_k < 1
    res_low_k = client.post("/predict", json={"text": "valid prompt", "top_k": 0})
    assert res_low_k.status_code == 422

    # top_k > 50
    res_high_k = client.post("/predict", json={"text": "valid prompt", "top_k": 51})
    assert res_high_k.status_code == 422


def test_generate_greedy_determinism(client):
    """Test that greedy decoding is deterministic across repeated calls."""
    payload = {"prompt": "In the year", "max_new_tokens": 10, "greedy": True}
    res1 = client.post("/generate", json=payload)
    res2 = client.post("/generate", json=payload)
    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json()["generated_text"] == res2.json()["generated_text"]


def test_generate_sampling_controls(client):
    """Test generation with temperature, Top-K, and Top-P controls."""
    payload = {
        "prompt": "Scientific research shows that",
        "max_new_tokens": 12,
        "temperature": 0.7,
        "top_k": 25,
        "top_p": 0.85,
        "greedy": False,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tokens_generated"] > 0
    assert data["finish_reason"] in ["length", "eos"]


def test_generate_max_new_tokens_bound(client):
    """Test that max_new_tokens is strictly respected."""
    for n in [3, 7, 15]:
        response = client.post(
            "/generate",
            json={"prompt": "Data science and", "max_new_tokens": n, "greedy": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tokens_generated"] <= n


def test_generate_invalid_parameters(client):
    """Test validation errors for /generate."""
    # Empty prompt
    res_empty = client.post("/generate", json={"prompt": "  ", "max_new_tokens": 10})
    assert res_empty.status_code in [400, 422]

    # max_new_tokens < 1
    res_zero_tokens = client.post("/generate", json={"prompt": "valid", "max_new_tokens": 0})
    assert res_zero_tokens.status_code == 422

    # max_new_tokens > 250
    res_huge_tokens = client.post("/generate", json={"prompt": "valid", "max_new_tokens": 300})
    assert res_huge_tokens.status_code == 422

    # temperature out of range
    res_neg_temp = client.post("/generate", json={"prompt": "valid", "temperature": -0.5})
    assert res_neg_temp.status_code == 422


def test_unmanaged_client_auto_initialization():
    """Verify that TestClient without lifespan context manager auto-initializes seamlessly."""
    unmanaged_client = TestClient(app)
    response = unmanaged_client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"

    pred_res = unmanaged_client.post("/predict", json={"text": "Machine learning", "top_k": 3})
    assert pred_res.status_code == 200
    assert len(pred_res.json()["predictions"]) == 3


def test_sampling_multinomial_zero_sum_and_inf_safety():
    """Verify that sample_next_token does not crash on non-finite or zero-sum probabilities."""
    # Tensor with all -inf logits
    all_inf_logits = torch.full((50,), -float("inf"))
    token = sample_next_token(all_inf_logits, temperature=1.0)
    assert 0 <= token < 50

    # Tensor with NaN logits
    nan_logits = torch.full((50,), float("nan"))
    token_nan = sample_next_token(nan_logits, temperature=1.0)
    assert isinstance(token_nan, int)


def test_lstm_default_vocab_size_matches_config():
    """Verify that LSTMLanguageModel default vocab_size matches ModelConfig and PRD specification."""
    cfg = ModelConfig()
    model = LSTMLanguageModel()
    assert model.vocab_size == cfg.vocab_size == 30000
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 26748208


def test_training_config_weight_decay():
    """Verify TrainingConfig.weight_decay is set to 0.01 per architecture and PRD specification."""
    from backend.model.config import TrainingConfig
    t_cfg = TrainingConfig()
    assert t_cfg.weight_decay == 0.01


def test_predict_with_unicode_symbols_and_emoji(client):
    """Verify /predict handles Unicode text and emojis without crashing."""
    response = client.post("/predict", json={"text": "Hello world 😊 🚀", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 3


def test_sample_next_token_nan_does_not_return_nan_index():
    """Verify sample_next_token does not select the index of NaN logits."""
    # Index 1 is NaN, but Index 0 is a strong valid logit
    logits = torch.tensor([50.0, float("nan"), 10.0])
    token_greedy = sample_next_token(logits, greedy=True)
    assert token_greedy == 0, f"Expected argmax to select index 0, but got index {token_greedy}"

    token_sampled = sample_next_token(logits, temperature=0.8)
    assert token_sampled != 1, f"Expected sampled token to avoid NaN index 1, but got {token_sampled}"


def test_concurrent_ensure_initialized():
    """Verify thread-safe double-checked initialization under concurrent thread access."""
    import concurrent.futures
    from backend.app.main import service

    errors = []

    def call_ensure():
        try:
            service.ensure_initialized()
            assert service.model is not None
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(call_ensure) for _ in range(16)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent initialization errors: {errors}"


def test_train_cli_default_weight_decay():
    """Verify CLI parser in train.py defaults weight_decay to 0.01 per PRD specification."""
    import argparse
    from backend.training.train import main
    # Parse --help or inspect parser defaults in train.py module
    import inspect
    from backend.training import train

    # Check argument parser default by inspecting parser in main
    # Or creating parser directly
    parser = argparse.ArgumentParser()
    # Check default specified in source of train.py
    with open("backend/training/train.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert 'default=0.01' in content
    assert 'default=1e-5' not in content



