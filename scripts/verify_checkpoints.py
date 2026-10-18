"""Unified verification script for LSTM Language Model checkpoints and inference compatibility.

Verifies:
1. Presence of best_val.pt and latest.pt.
2. Checkpoint metadata integrity (epoch, validation loss, architecture config).
3. State dict contains zero "module." prefixes.
4. Checkpoint loads cleanly into standard non-DDP LSTMLanguageModel on CPU.
5. Model loads and evaluates on CUDA (if available).
6. Existing FastAPI inference service can initialize and load best_val.pt.
7. Next-word prediction endpoint generates valid ranked token candidates.
"""

import argparse
import json
from pathlib import Path
import sys
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.model.lstm import LSTMLanguageModel
from backend.app.main import service, PredictRequest, predict_next_word


def verify_checkpoints(checkpoint_dir: Path):
    print("=" * 70)
    print("STARTING LSTM CHECKPOINT INTEGRITY & INFERENCE VERIFICATION")
    print("=" * 70)

    best_val_path = checkpoint_dir / "best_val.pt"
    latest_path = checkpoint_dir / "latest.pt"
    adapted_path = root_dir / "experiments" / "multidomain_v1" / "checkpoints" / "multidomain_adapted.pt"

    print(f"Checking artifacts in: {checkpoint_dir}")
    assert best_val_path.exists(), f"Missing required inference checkpoint: {best_val_path}!"
    print(f"Found best_val.pt ({best_val_path.stat().st_size / (1024**2):.1f} MB) [Primary Inference]")

    ckpts_to_verify = [("best_val.pt", best_val_path)]
    if latest_path.exists():
        print(f"Found latest.pt ({latest_path.stat().st_size / (1024**2):.1f} MB) [Training Continuity]")
        ckpts_to_verify.append(("latest.pt", latest_path))
    else:
        print("Note: latest.pt not present in package (optional external training continuity checkpoint).")

    if adapted_path.exists():
        print(f"Found multidomain_adapted.pt ({adapted_path.stat().st_size / (1024**2):.1f} MB) [Domain Adaptation]")
        ckpts_to_verify.append(("multidomain_adapted.pt", adapted_path))

    for name, path in ckpts_to_verify:
        print("-" * 50)
        print(f"Verifying {name}...")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)

        epoch = ckpt.get("epoch")
        val_loss = ckpt.get("val_loss")
        print(f"[{name}] Reported Epoch: {epoch}, Val Loss: {val_loss:.4f}" if val_loss is not None else f"[{name}] Reported Epoch: {epoch}")
        assert epoch is not None and epoch > 0, f"Invalid epoch in {name}: {epoch}"
        print(f"  -> PASSED: Epoch correctly recorded ({epoch}).")

        # Check zero "module." prefixes
        state_dict = ckpt.get("model_state_dict", {})
        assert len(state_dict) > 0, f"Empty state dict in {name}!"
        module_keys = [k for k in state_dict.keys() if k.startswith("module.")]
        assert len(module_keys) == 0, f"Found 'module.' prefix in {name}: {module_keys[:3]}"
        print(f"  -> PASSED: Zero 'module.' prefixes found across {len(state_dict)} tensors.")

        # Check non-DDP model on CPU
        cfg = ckpt.get("config", {})
        model_cpu = LSTMLanguageModel(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            pad_idx=cfg.get("pad_idx", 0),
        )
        model_cpu.load_state_dict(state_dict)
        model_cpu.eval()
        dummy_cpu = torch.randint(0, cfg["vocab_size"], (1, 35))
        with torch.no_grad():
            out_cpu, _ = model_cpu(dummy_cpu)
        assert out_cpu.shape == (1, 35, cfg["vocab_size"])
        print(f"  -> PASSED: Loaded and evaluated in non-DDP LSTMLanguageModel on CPU.")

        # Check non-DDP model on CUDA
        if torch.cuda.is_available():
            model_cuda = LSTMLanguageModel(
                vocab_size=cfg["vocab_size"],
                embed_dim=cfg["embed_dim"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                pad_idx=cfg.get("pad_idx", 0),
            ).cuda()
            model_cuda.load_state_dict(state_dict)
            model_cuda.eval()
            dummy_cuda = torch.randint(0, cfg["vocab_size"], (1, 35)).cuda()
            with torch.no_grad():
                out_cuda, _ = model_cuda(dummy_cuda)
            assert out_cuda.shape == (1, 35, cfg["vocab_size"])
            print(f"  -> PASSED: Loaded and evaluated in non-DDP LSTMLanguageModel on CUDA ({torch.cuda.get_device_name(0)}).")

    # Check FastAPI inference service integration with best_val.pt
    print("-" * 50)
    print("Testing FastAPI InferenceService integration with best_val.pt...")
    service.initialize()
    assert service.checkpoint_loaded, "FastAPI InferenceService failed to load checkpoint!"
    assert service.model is not None, "Model not initialized in InferenceService!"
    print(f"InferenceService initialized successfully (Device: {service.device}, Checkpoint Loaded: {service.checkpoint_loaded}).")

    # 1. Autocomplete Next-Word Prediction
    prompt = "The history of the world"
    req = PredictRequest(text=prompt, top_k=5)
    res = predict_next_word(req)
    print(f"Test prompt: '{prompt}'")
    top_cand = res.predictions[0] if res.predictions else None
    print(f"Top Candidate: '{top_cand.token}' (Prob: {top_cand.probability:.4f})")
    print(f"All candidates: {[(c.token, round(c.probability, 4)) for c in res.predictions]}")
    assert len(res.predictions) == 5, f"Expected 5 candidates, got {len(res.predictions)}!"
    assert all(0.0 <= c.probability <= 1.0 for c in res.predictions), "Invalid probability range!"
    # Ensure no structural tokens in predictions
    structural_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
    assert all(c.token not in structural_tokens for c in res.predictions), "Structural tokens found in predictions!"
    print("  -> PASSED: Autocomplete predict_next_word produced 5 valid candidates without structural tokens.")

    # 2. Autoregressive Text Generation — Greedy Decoding (Determinism)
    from backend.app.main import GenerateRequest, generate_text
    gen_req_greedy = GenerateRequest(prompt="In the year", max_new_tokens=10, greedy=True)
    gen_res1 = generate_text(gen_req_greedy)
    gen_res2 = generate_text(gen_req_greedy)
    assert gen_res1.generated_text == gen_res2.generated_text, "Greedy decoding is not deterministic!"
    assert gen_res1.tokens_generated > 0, "Zero tokens generated in greedy mode!"
    print(f"  -> PASSED: Greedy generation deterministic: '{gen_res1.generated_text}' ({gen_res1.tokens_generated} tokens).")

    # 3. Autoregressive Text Generation — Sampling (Temperature, Top-K, Top-P)
    gen_req_sample = GenerateRequest(
        prompt="Scientific research shows that",
        max_new_tokens=15,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        greedy=False,
    )
    gen_res_sample = generate_text(gen_req_sample)
    assert gen_res_sample.tokens_generated > 0, "Zero tokens generated in sampling mode!"
    print(f"  -> PASSED: Sampled generation: '{gen_res_sample.generated_text}' ({gen_res_sample.tokens_generated} tokens).")

    # 4. Edge Cases: Unicode, Emojis, Multilingual, Context-Exceeding Prompts
    unicode_prompt = "Deep learning 🚀🤖 and 机器学习"
    res_unicode = predict_next_word(PredictRequest(text=unicode_prompt, top_k=3))
    assert len(res_unicode.predictions) == 3, "Failed to handle unicode/emoji prompt!"
    gen_unicode = generate_text(GenerateRequest(prompt=unicode_prompt, max_new_tokens=5, greedy=True))
    assert gen_unicode.tokens_generated > 0, "Failed to generate from unicode/emoji prompt!"
    print("  -> PASSED: Unicode/emoji input handled safely without tokenizer crash.")

    # Context window exceeding prompt (>35 tokens)
    long_prompt = "The quick brown fox jumps over the lazy dog and runs across the wide open fields " * 5
    res_long = predict_next_word(PredictRequest(text=long_prompt, top_k=5))
    assert len(res_long.predictions) == 5, "Failed on prompt exceeding context length!"
    print("  -> PASSED: Context length truncation handled correctly (>35 tokens).")

    print("=" * 70)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify LSTM checkpoints and inference service.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="backend/model_artifacts/checkpoints",
        help="Directory containing best_val.pt and latest.pt",
    )
    args = parser.parse_args()
    verify_checkpoints(Path(args.checkpoint_dir))
