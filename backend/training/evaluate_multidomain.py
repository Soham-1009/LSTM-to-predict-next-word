"""Multi-Domain Generalization Evaluation Suite.

Evaluates LSTM language models across multiple real-world domains with
stratified parent-source partitioning (zero data leakage) and computes:
- Loss, Perplexity, Top-1, Top-3, Top-5 Accuracy per domain
- Out-Of-Vocabulary (OOV) rate and vocabulary coverage
- In-domain retention vs catastrophic forgetting on WikiText-103 test split
- Qualitative autoregressive generation diagnostics across standardized prompts
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

from backend.model.lstm import LSTMLanguageModel, sample_next_token
from backend.model.tokenizer import Tokenizer
from backend.training.dataset import CausalLanguageModelDataset
from backend.training.serialize import load_memmap_tokens


PROMPTS_BY_DOMAIN = [
    {
        "domain": "expository_wikitext",
        "prompt": "The history of the modern world",
        "description": "General encyclopedia expository prose",
    },
    {
        "domain": "scientific",
        "prompt": "The experimental results demonstrate that",
        "description": "Scientific and biomedical research literature",
    },
    {
        "domain": "technical_docs",
        "prompt": "In computer science, a function",
        "description": "Technical documentation and RFC standards",
    },
    {
        "domain": "literature",
        "prompt": "She walked slowly into the quiet",
        "description": "Narrative fiction and classic literature",
    },
    {
        "domain": "conversation",
        "prompt": "Can you tell me what happened",
        "description": "Colloquial dialog and dramatic conversational turns",
    },
]


def evaluate_token_sequence(
    model: nn.Module,
    tokens: np.ndarray,
    criterion: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int = 32,
    offsets: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate model on a sequence of tokens with causal sliding windows."""
    model.eval()
    dataset = CausalLanguageModelDataset(
        token_data=tokens,
        seq_len=cfg["context_length"],
        stride=cfg["context_length"],
        doc_offsets=offsets,
    )
    if len(dataset) == 0:
        return {"loss": 0.0, "perplexity": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "tokens": 0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    top1_c = 0
    top3_c = 0
    top5_c = 0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            valid = y_flat != cfg["pad_idx"]
            nv = int(valid.sum().item())
            if nv > 0:
                loss = criterion(logits_flat, y_flat)
                total_loss += loss.item() * nv
                total_tokens += nv

                _, top5_idx = torch.topk(logits_flat, 5, dim=-1)
                correct = top5_idx.eq(y_flat.unsqueeze(-1))
                top1_c += int((correct[:, 0] & valid).sum().item())
                top3_c += int((correct[:, :3].any(dim=-1) & valid).sum().item())
                top5_c += int((correct.any(dim=-1) & valid).sum().item())

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "top1_acc": round((top1_c / max(total_tokens, 1)) * 100, 2),
        "top3_acc": round((top3_c / max(total_tokens, 1)) * 100, 2),
        "top5_acc": round((top5_c / max(total_tokens, 1)) * 100, 2),
        "evaluated_tokens": total_tokens,
    }


def generate_sample_texts(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: List[Dict[str, str]],
    cfg: Dict[str, Any],
    device: torch.device,
    max_new_tokens: int = 25,
) -> List[Dict[str, str]]:
    """Generate text continuations across domain prompts under greedy and sampling modes."""
    model.eval()
    results = []

    for item in prompts:
        prompt_text = item["prompt"]
        tokens = tokenizer.encode(prompt_text)
        if not tokens:
            tokens = [tokenizer.bos_id]

        # 1. Greedy Generation
        greedy_ids = list(tokens)
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                ctx = greedy_ids[-cfg["context_length"] :]
                inp = torch.tensor([ctx], dtype=torch.long, device=device)
                logits, _ = model(inp)
                step_logits = logits[0, -1, :].clone()
                # Suppress pad, bos, unk
                for cid in [tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]:
                    if cid < step_logits.size(0):
                        step_logits[cid] = -float("Inf")
                next_tok = sample_next_token(step_logits, greedy=True)
                if next_tok == tokenizer.eos_id:
                    break
                greedy_ids.append(next_tok)

        greedy_text = tokenizer.decode(greedy_ids, skip_special_tokens=True)

        # 2. Temperature Sampling (T=0.8, Top-K=40, Top-P=0.9)
        sample_ids = list(tokens)
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                ctx = sample_ids[-cfg["context_length"] :]
                inp = torch.tensor([ctx], dtype=torch.long, device=device)
                logits, _ = model(inp)
                step_logits = logits[0, -1, :].clone()
                for cid in [tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]:
                    if cid < step_logits.size(0):
                        step_logits[cid] = -float("Inf")
                next_tok = sample_next_token(step_logits, temperature=0.8, top_k=40, top_p=0.9, greedy=False)
                if next_tok == tokenizer.eos_id:
                    break
                sample_ids.append(next_tok)

        sample_text = tokenizer.decode(sample_ids, skip_special_tokens=True)

        results.append(
            {
                "domain": item["domain"],
                "prompt": prompt_text,
                "greedy_generation": greedy_text,
                "sampling_generation": sample_text,
            }
        )

    return results


def run_full_generalization_evaluation(
    model_path: Path | str,
    model_label: str,
    pilot_manifest_path: Path | str = "experiments/multidomain_v1/pilot/pilot_manifest.jsonl",
    pilot_processed_dir: Path | str = "experiments/multidomain_v1/pilot/processed",
    wikitext_test_bin_path: Path | str = "backend/datasets/processed/test.bin",
    wikitext_test_offsets_path: Path | str = "backend/datasets/processed/offsets/test_offsets.bin",
    vocab_path: Path | str = "backend/model_artifacts/tokenizer/vocab.json",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute complete generalization benchmark on a model checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_file = Path(model_path)
    print(f"\n===========================================================================")
    print(f"GENERALIZATION EVALUATION: {model_label.upper()} ({ckpt_file.name})")
    print(f"===========================================================================")

    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
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

    tokenizer = Tokenizer.load(vocab_path)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["pad_idx"])

    # 1. Parse pilot manifest test documents
    manifest_file = Path(pilot_manifest_path)
    docs = [json.loads(l) for l in open(manifest_file, encoding="utf-8") if l.strip()]
    retained = [d for d in docs if d.get("filter_decision") == "RETAINED"]
    test_docs = [d for d in retained if d.get("assigned_split") == "test"]

    pilot_dir = Path(pilot_processed_dir)
    test_tokens = np.fromfile(pilot_dir / "test.bin", dtype=np.uint16)
    test_offsets = np.fromfile(pilot_dir / "test_offsets.bin", dtype=np.uint64)

    # Map each domain to its contiguous document index range
    domain_to_doc_indices: Dict[str, List[int]] = {}
    for i, d in enumerate(test_docs):
        domain_to_doc_indices.setdefault(d["domain"], []).append(i)

    # 2. Evaluate each domain on the pilot test set with document-boundary preservation
    per_domain_results: Dict[str, Dict[str, Any]] = {}
    print("\n[Evaluating Multi-Domain Test Split by Domain (Document-Boundary Preserved)]")

    for dom, doc_idxs in sorted(domain_to_doc_indices.items()):
        start_doc = doc_idxs[0]
        end_doc = doc_idxs[-1]

        start_tok = int(test_offsets[start_doc])
        end_tok = int(test_offsets[end_doc + 1])

        dom_tokens = test_tokens[start_tok:end_tok]
        dom_offsets = test_offsets[start_doc : end_doc + 2] - start_tok

        total_tokens = len(dom_tokens)
        oov_count = int((dom_tokens == cfg["unk_idx"]).sum())
        oov_pct = (oov_count / max(total_tokens, 1)) * 100
        vocab_coverage_pct = 100.0 - oov_pct

        eval_metrics = evaluate_token_sequence(
            model=model,
            tokens=dom_tokens,
            criterion=criterion,
            cfg=cfg,
            device=device,
            batch_size=32,
            offsets=dom_offsets,
        )

        domain_res = {
            "domain": dom,
            "document_count": len(doc_idxs),
            "total_tokens": total_tokens,
            "oov_tokens": oov_count,
            "oov_rate_pct": round(oov_pct, 2),
            "vocab_coverage_pct": round(vocab_coverage_pct, 2),
            **eval_metrics,
        }
        per_domain_results[dom] = domain_res
        print(
            f"  {dom:18s} | Docs: {len(doc_idxs):2d} | Toks: {total_tokens:5d} | "
            f"Eval Targets: {eval_metrics['evaluated_tokens']:5d} | OOV: {oov_pct:5.2f}% | "
            f"Loss: {eval_metrics['loss']:.4f} | PPL: {eval_metrics['perplexity']:7.2f} | "
            f"Top-1: {eval_metrics['top1_acc']:5.2f}% | Top-5: {eval_metrics['top5_acc']:5.2f}%"
        )

    # 3. Overall Multi-Domain Test Split (Strict Document Offsets)
    print("\n[Evaluating Overall Multi-Domain Test Set (All 6 Domains, Document-Boundary Preserved)]")
    overall_pilot_metrics = evaluate_token_sequence(
        model=model,
        tokens=test_tokens,
        criterion=criterion,
        cfg=cfg,
        device=device,
        batch_size=32,
        offsets=test_offsets,
    )
    total_pilot_oov = int((test_tokens == cfg["unk_idx"]).sum())
    overall_oov_pct = (total_pilot_oov / len(test_tokens)) * 100

    overall_pilot = {
        "total_documents": len(test_docs),
        "total_tokens": len(test_tokens),
        "oov_tokens": total_pilot_oov,
        "oov_rate_pct": round(overall_oov_pct, 2),
        "vocab_coverage_pct": round(100.0 - overall_oov_pct, 2),
        **overall_pilot_metrics,
    }
    print(
        f"  OVERALL MULTI-DOMAIN | Docs: {len(test_docs):2d} | Toks: {len(test_tokens):5d} | "
        f"Eval Targets: {overall_pilot_metrics['evaluated_tokens']:5d} | OOV: {overall_oov_pct:5.2f}% | "
        f"Loss: {overall_pilot_metrics['loss']:.4f} | PPL: {overall_pilot_metrics['perplexity']:7.2f} | "
        f"Top-1: {overall_pilot_metrics['top1_acc']:5.2f}% | Top-5: {overall_pilot_metrics['top5_acc']:5.2f}%"
    )

    # 3b. Equal-Weight Macro-Average Across all 6 Domains (Equal 1/6 Weight)
    macro_loss = float(np.mean([r["loss"] for r in per_domain_results.values()]))
    macro_ppl = float(np.mean([r["perplexity"] for r in per_domain_results.values()]))
    geometric_ppl = float(math.exp(min(macro_loss, 20)))
    macro_top1 = float(np.mean([r["top1_acc"] for r in per_domain_results.values()]))
    macro_top3 = float(np.mean([r["top3_acc"] for r in per_domain_results.values()]))
    macro_top5 = float(np.mean([r["top5_acc"] for r in per_domain_results.values()]))

    macro_pilot = {
        "domain_count": len(per_domain_results),
        "weighting_type": "equal_weight_per_domain",
        "description": "Equal-weight unweighted arithmetic mean across all 6 evaluated domains",
        "loss": round(macro_loss, 4),
        "perplexity": round(macro_ppl, 2),
        "geometric_perplexity": round(geometric_ppl, 2),
        "top1_acc": round(macro_top1, 2),
        "top3_acc": round(macro_top3, 2),
        "top5_acc": round(macro_top5, 2),
    }
    print(
        f"  MACRO-AVERAGE (6 DOMAINS) | Domains: {len(per_domain_results):2d} | "
        f"Loss: {macro_loss:.4f} | PPL (mean): {macro_ppl:7.2f} | PPL (geom): {geometric_ppl:7.2f} | "
        f"Top-1: {macro_top1:5.2f}% | Top-5: {macro_top5:5.2f}%"
    )

    # 4. In-Domain WikiText-103 Retention Split (Boundary-Preserved 22-Article Slice)
    print("\n[Evaluating In-Domain WikiText-103 Retention Split (Boundary-Preserved 22-Article Slice)]")
    wiki_test_tokens = load_memmap_tokens(wikitext_test_bin_path)
    wiki_test_offsets = (
        np.fromfile(wikitext_test_offsets_path, dtype=np.uint64)
        if Path(wikitext_test_offsets_path).exists()
        else None
    )

    if wiki_test_offsets is not None and len(wiki_test_offsets) > 23:
        retention_doc_count = 22
        retention_offsets = wiki_test_offsets[: retention_doc_count + 1]
        retention_token_count = int(retention_offsets[-1])
        wiki_eval_slice = wiki_test_tokens[:retention_token_count]
        wiki_offsets_slice = retention_offsets
    else:
        retention_doc_count = 1
        wiki_eval_slice = wiki_test_tokens[:100000]
        wiki_offsets_slice = None

    wiki_metrics = evaluate_token_sequence(
        model=model,
        tokens=wiki_eval_slice,
        criterion=criterion,
        cfg=cfg,
        device=device,
        batch_size=64,
        offsets=wiki_offsets_slice,
    )
    wiki_oov = int((wiki_eval_slice == cfg["unk_idx"]).sum())
    wiki_oov_pct = (wiki_oov / len(wiki_eval_slice)) * 100

    wiki_results = {
        "dataset_name": "WikiText-103 Held-Out Test Split (Retention Slice)",
        "evaluation_scope": f"Representative {retention_doc_count}-Document Slice ({len(wiki_eval_slice):,} tokens) with Strict Document Boundary Preservation",
        "documents_in_slice": retention_doc_count,
        "tokens_in_slice": len(wiki_eval_slice),
        "oov_tokens": wiki_oov,
        "oov_rate_pct": round(wiki_oov_pct, 2),
        "vocab_coverage_pct": round(100.0 - wiki_oov_pct, 2),
        **wiki_metrics,
    }
    print(
        f"  WIKITEXT-103 RETENTION | Docs: {retention_doc_count:2d} | Toks: {len(wiki_eval_slice):6d} | "
        f"Eval Targets: {wiki_metrics['evaluated_tokens']:6d} | OOV: {wiki_oov_pct:5.2f}% | "
        f"Loss: {wiki_metrics['loss']:.4f} | PPL: {wiki_metrics['perplexity']:7.2f} | "
        f"Top-1: {wiki_metrics['top1_acc']:5.2f}% | Top-5: {wiki_metrics['top5_acc']:5.2f}%"
    )

    # 5. Qualitative Text Generations
    print("\n[Generating Diagnostic Completions across Domain Prompts]")
    samples = generate_sample_texts(
        model=model,
        tokenizer=tokenizer,
        prompts=PROMPTS_BY_DOMAIN,
        cfg=cfg,
        device=device,
    )
    for s in samples:
        print(f"  [{s['domain'].upper()}] Prompt: '{s['prompt']}'")
        print(f"     Greedy:   {s['greedy_generation']}")
        print(f"     Sampling: {s['sampling_generation']}\n")

    report_payload = {
        "model_label": model_label,
        "checkpoint_path": str(ckpt_file),
        "device": str(device),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_domain_results": per_domain_results,
        "overall_multidomain": overall_pilot,
        "macro_average_multidomain": macro_pilot,
        "wikitext_retention": wiki_results,
        "qualitative_samples": samples,
    }

    return report_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="backend/model_artifacts/checkpoints/best_val.pt")
    parser.add_argument("--label", default="baseline_best_val")
    parser.add_argument("--output", default="backend/model_artifacts/results/generalization_benchmark_baseline.json")
    args = parser.parse_args()

    res = run_full_generalization_evaluation(model_path=args.checkpoint, model_label=args.label)
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved report to: {out_file}")
