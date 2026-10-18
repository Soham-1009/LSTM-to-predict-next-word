"""Authoritative Multi-Domain Pilot Artifact Audit Routine.

Validates that pilot_manifest.jsonl, processed/metadata.json, and the 6 binary
files (train.bin, val.bin, test.bin, train_offsets.bin, val_offsets.bin, test_offsets.bin)
are 100% reconciled and mathematically consistent.

Enforces:
1. Raw records = retained records + duplicate records + rejected records.
2. Retained record count equals unique_document_count in metadata.
3. Every retained record belongs to exactly one valid split (train, val, test).
4. For each split, sum of document token counts equals binary token count.
5. Each offset file contains exactly document_count + 1 offsets.
6. Offsets start at 0, strictly increase for non-empty documents, and end at total tokens.
7. Binary file size matches token count * dtype size.
8. Recomputed SHA-256 hashes match metadata.
9. Domain breakdown in metadata reconciles exactly with manifest.
10. Domain breakdown sums reproduce split and total statistics.
11. Rejected/duplicate records never contribute to retained token counts.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_pilot_artifacts(pilot_dir: Path) -> Dict[str, Any]:
    """Execute authoritative mathematical reconciliation across pilot artifacts."""
    manifest_path = pilot_dir / "pilot_manifest.jsonl"
    processed_dir = pilot_dir / "processed"
    metadata_path = processed_dir / "metadata.json"

    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"
    assert metadata_path.exists(), f"Missing metadata at {metadata_path}"

    # 1. Parse manifest
    manifest_records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_records.append(json.loads(line))

    retained_records = [r for r in manifest_records if r.get("filter_decision") == "RETAINED"]
    duplicate_records = [r for r in manifest_records if r.get("filter_decision") == "REJECTED_EXACT_DUPLICATE"]
    rejected_records = [r for r in manifest_records if r.get("filter_decision") == "REJECTED_QUALITY_FILTER"]

    # Assertion 1: Raw = retained + duplicate + rejected
    total_raw = len(manifest_records)
    assert total_raw == len(retained_records) + len(duplicate_records) + len(rejected_records), (
        f"Raw count mismatch: total={total_raw} vs sum={len(retained_records) + len(duplicate_records) + len(rejected_records)}"
    )

    # Assertion 11: Deduplicated and rejected records never contribute tokens
    for d in duplicate_records:
        assert d.get("token_count", 0) == 0, f"Duplicate record {d['doc_id']} has non-zero tokens!"
        assert d.get("assigned_split") == "DEDUPLICATED", f"Unexpected split for duplicate: {d.get('assigned_split')}"
    for r in rejected_records:
        assert r.get("token_count", 0) == 0, f"Rejected record {r['doc_id']} has non-zero tokens!"
        assert r.get("assigned_split") == "REJECTED", f"Unexpected split for rejected: {r.get('assigned_split')}"

    # 2. Parse metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    stats = metadata["statistics"]
    # Assertion 2: Retained count equals metadata unique_document_count
    assert len(retained_records) == stats["unique_document_count"], (
        f"Retained record count ({len(retained_records)}) != unique_document_count ({stats['unique_document_count']})"
    )
    assert total_raw == stats["raw_document_count"], (
        f"Manifest raw count ({total_raw}) != metadata raw count ({stats['raw_document_count']})"
    )
    assert len(duplicate_records) == stats["duplicate_document_count"], (
        f"Manifest duplicate count ({len(duplicate_records)}) != metadata duplicate count ({stats['duplicate_document_count']})"
    )
    assert len(rejected_records) == stats["rejected_document_count"], (
        f"Manifest rejected count ({len(rejected_records)}) != metadata rejected count ({stats['rejected_document_count']})"
    )

    # 3. Split membership and disjointness
    valid_splits = {"train", "val", "test"}
    split_docs: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    parent_to_split: Dict[str, str] = {}
    parent_sources_by_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    for r in retained_records:
        split = r.get("assigned_split")
        # Assertion 3: Every retained record belongs to exactly one valid split
        assert split in valid_splits, f"Invalid split '{split}' in record {r['doc_id']}"
        split_docs[split].append(r)

        parent_id = r["parent_source_id"]
        if parent_id in parent_to_split:
            assert parent_to_split[parent_id] == split, (
                f"PARENT LEAKAGE! Parent {parent_id} found in both {parent_to_split[parent_id]} and {split}"
            )
        else:
            parent_to_split[parent_id] = split
        parent_sources_by_split[split].add(parent_id)

    # Verify zero parent overlap
    assert len(parent_sources_by_split["train"].intersection(parent_sources_by_split["val"])) == 0
    assert len(parent_sources_by_split["train"].intersection(parent_sources_by_split["test"])) == 0
    assert len(parent_sources_by_split["val"].intersection(parent_sources_by_split["test"])) == 0

    # 4. Binary and Offset Reconciliation
    dtype_str = metadata.get("token_dtype", "uint16")
    token_dtype = np.uint16 if dtype_str == "uint16" else np.uint32
    dtype_itemsize = np.dtype(token_dtype).itemsize

    total_tokens_across_splits = 0
    recomputed_checksums: Dict[str, str] = {}

    for split in ["train", "val", "test"]:
        bin_file = processed_dir / f"{split}.bin"
        offsets_file = processed_dir / f"{split}_offsets.bin"

        assert bin_file.exists(), f"Binary file not found: {bin_file}"
        assert offsets_file.exists(), f"Offsets file not found: {offsets_file}"

        # Assertion 8: Recalculated SHA-256 matches metadata
        bin_sha = compute_sha256(bin_file)
        offsets_sha = compute_sha256(offsets_file)
        recomputed_checksums[f"{split}.bin"] = bin_sha
        recomputed_checksums[f"{split}_offsets.bin"] = offsets_sha

        assert bin_sha == metadata["file_checksums"][f"{split}.bin"], (
            f"Checksum mismatch for {split}.bin: computed {bin_sha} vs metadata {metadata['file_checksums'][f'{split}.bin']}"
        )
        assert offsets_sha == metadata["file_checksums"][f"{split}_offsets.bin"], (
            f"Checksum mismatch for {split}_offsets.bin: computed {offsets_sha} vs metadata {metadata['file_checksums'][f'{split}_offsets.bin']}"
        )

        tokens = np.fromfile(bin_file, dtype=token_dtype)
        offsets = np.fromfile(offsets_file, dtype=np.uint64)
        doc_list = split_docs[split]

        # Assertion 4: Sum of document tokens in manifest equals binary token count
        manifest_tok_sum = sum(d["token_count"] for d in doc_list)
        assert len(tokens) == manifest_tok_sum, (
            f"Token count mismatch for {split}: binary has {len(tokens)} vs manifest sum {manifest_tok_sum}"
        )
        assert len(tokens) == stats[split]["tokens"], (
            f"Token count mismatch for {split}: binary has {len(tokens)} vs metadata stats {stats[split]['tokens']}"
        )

        # Assertion 5: Each offset file contains exactly document_count + 1 offsets
        assert len(offsets) == len(doc_list) + 1, (
            f"Offset count mismatch for {split}: {len(offsets)} offsets vs {len(doc_list)} docs + 1"
        )
        assert len(offsets) == stats[split]["offsets_count"], (
            f"Offset count mismatch for {split}: {len(offsets)} offsets vs metadata {stats[split]['offsets_count']}"
        )

        # Assertion 6: Offsets start at 0, strictly increase, and final offset equals token count
        assert offsets[0] == 0, f"First offset in {split}_offsets.bin is not 0 ({offsets[0]})"
        assert offsets[-1] == len(tokens), (
            f"Final offset in {split}_offsets.bin ({offsets[-1]}) != token count ({len(tokens)})"
        )
        for i in range(len(offsets) - 1):
            doc_len = int(offsets[i + 1] - offsets[i])
            expected_len = doc_list[i]["token_count"]
            assert doc_len == expected_len, (
                f"Offset interval {i} length ({doc_len}) != manifest token count ({expected_len}) in {split}"
            )
            if expected_len > 0:
                assert offsets[i + 1] > offsets[i], f"Non-strictly increasing offset at doc {i} in {split}"

        # Assertion 7: Binary file size matches token count * dtype size
        expected_bytes = len(tokens) * dtype_itemsize
        actual_bytes = bin_file.stat().st_size
        assert actual_bytes == expected_bytes, (
            f"File size mismatch for {split}.bin: actual {actual_bytes} != expected {expected_bytes}"
        )
        assert actual_bytes == stats[split]["bytes"], (
            f"File size mismatch for {split}.bin: actual {actual_bytes} != metadata {stats[split]['bytes']}"
        )

        # Verify parent count in stats matches unique parents in split
        assert len(parent_sources_by_split[split]) == stats[split]["parents"], (
            f"Parent count mismatch for {split}: actual {len(parent_sources_by_split[split])} != metadata {stats[split]['parents']}"
        )
        assert len(doc_list) == stats[split]["documents"], (
            f"Doc count mismatch for {split}: actual {len(doc_list)} != metadata {stats[split]['documents']}"
        )

        total_tokens_across_splits += len(tokens)

    assert total_tokens_across_splits == stats["total_tokens"], (
        f"Total tokens across splits ({total_tokens_across_splits}) != metadata total ({stats['total_tokens']})"
    )

    # 5. Domain Breakdown Reconciliation (Assertions 9 & 10)
    domain_breakdown = metadata["domain_breakdown"]
    domains_in_manifest = {r["domain"] for r in retained_records}
    assert domains_in_manifest == set(domain_breakdown.keys()), (
        f"Domain set mismatch: manifest={domains_in_manifest} vs metadata={set(domain_breakdown.keys())}"
    )

    domain_accumulators = {
        dom: {
            "parents": {"train": set(), "val": set(), "test": set()},
            "docs": {"train": 0, "val": 0, "test": 0, "total": 0},
            "tokens": {"train": 0, "val": 0, "test": 0, "total": 0},
            "unk_tokens": 0,
        }
        for dom in domain_breakdown.keys()
    }

    for r in retained_records:
        dom = r["domain"]
        split = r["assigned_split"]
        toks = r["token_count"]
        unks = r.get("oov_token_count", 0)
        pid = r["parent_source_id"]

        acc = domain_accumulators[dom]
        acc["parents"][split].add(pid)
        acc["docs"][split] += 1
        acc["docs"]["total"] += 1
        acc["tokens"][split] += toks
        acc["tokens"]["total"] += toks
        acc["unk_tokens"] += unks

    for dom, acc in domain_accumulators.items():
        meta_dom = domain_breakdown[dom]
        # Documents check
        for k in ["train", "val", "test", "total"]:
            assert acc["docs"][k] == meta_dom["documents"][k], (
                f"Domain '{dom}' docs['{k}'] mismatch: manifest={acc['docs'][k]} vs metadata={meta_dom['documents'][k]}"
            )
            assert acc["tokens"][k] == meta_dom["tokens"][k], (
                f"Domain '{dom}' tokens['{k}'] mismatch: manifest={acc['tokens'][k]} vs metadata={meta_dom['tokens'][k]}"
            )

        # Parents check
        for k in ["train", "val", "test"]:
            actual_p_count = len(acc["parents"][k])
            meta_p_count = meta_dom["parent_sources"][k]
            assert actual_p_count == meta_p_count, (
                f"Domain '{dom}' parent_sources['{k}'] mismatch: manifest={actual_p_count} vs metadata={meta_p_count}"
            )
        total_p = sum(len(acc["parents"][k]) for k in ["train", "val", "test"])
        assert total_p == meta_dom["parent_sources"]["total"], (
            f"Domain '{dom}' total parents mismatch: manifest={total_p} vs metadata={meta_dom['parent_sources']['total']}"
        )

        # UNK tokens check
        assert acc["unk_tokens"] == meta_dom["unk_tokens"], (
            f"Domain '{dom}' unk_tokens mismatch: manifest={acc['unk_tokens']} vs metadata={meta_dom['unk_tokens']}"
        )
        expected_oov = round(acc["unk_tokens"] / acc["tokens"]["total"] * 100, 2) if acc["tokens"]["total"] > 0 else 0.0
        assert expected_oov == meta_dom["oov_rate_pct"], (
            f"Domain '{dom}' oov_rate_pct mismatch: calculated {expected_oov} vs metadata {meta_dom['oov_rate_pct']}"
        )

    # Assertion 10: Summing across domain breakdowns reproduces split and total statistics
    sum_tr_tokens = sum(d["tokens"]["train"] for d in domain_breakdown.values())
    sum_val_tokens = sum(d["tokens"]["val"] for d in domain_breakdown.values())
    sum_te_tokens = sum(d["tokens"]["test"] for d in domain_breakdown.values())
    sum_tot_tokens = sum(d["tokens"]["total"] for d in domain_breakdown.values())

    assert sum_tr_tokens == stats["train"]["tokens"]
    assert sum_val_tokens == stats["val"]["tokens"]
    assert sum_te_tokens == stats["test"]["tokens"]
    assert sum_tot_tokens == stats["total_tokens"]

    sum_tr_docs = sum(d["documents"]["train"] for d in domain_breakdown.values())
    sum_val_docs = sum(d["documents"]["val"] for d in domain_breakdown.values())
    sum_te_docs = sum(d["documents"]["test"] for d in domain_breakdown.values())
    sum_tot_docs = sum(d["documents"]["total"] for d in domain_breakdown.values())

    assert sum_tr_docs == stats["train"]["documents"]
    assert sum_val_docs == stats["val"]["documents"]
    assert sum_te_docs == stats["test"]["documents"]
    assert sum_tot_docs == stats["unique_document_count"]

    sum_tot_parents = sum(d["parent_sources"]["total"] for d in domain_breakdown.values())
    assert sum_tot_parents == stats["total_parent_sources"]

    return {
        "status": "AUDIT_PASSED",
        "raw_record_count": total_raw,
        "retained_record_count": len(retained_records),
        "duplicate_record_count": len(duplicate_records),
        "rejected_record_count": len(rejected_records),
        "total_parent_sources": stats["total_parent_sources"],
        "total_tokens": stats["total_tokens"],
        "split_counts": {
            split: {
                "parents": stats[split]["parents"],
                "documents": stats[split]["documents"],
                "tokens": stats[split]["tokens"],
                "bytes": stats[split]["bytes"],
                "offsets": stats[split]["offsets_count"],
            }
            for split in ["train", "val", "test"]
        },
        "domain_summary": {
            dom: {
                "parents_tr_val_te": f"{meta['parent_sources']['train']}/{meta['parent_sources']['val']}/{meta['parent_sources']['test']}",
                "parents_total": meta["parent_sources"]["total"],
                "docs_tr_val_te": f"{meta['documents']['train']}/{meta['documents']['val']}/{meta['documents']['test']}",
                "docs_total": meta["documents"]["total"],
                "tokens_tr_val_te": f"{meta['tokens']['train']}/{meta['tokens']['val']}/{meta['tokens']['test']}",
                "tokens_total": meta["tokens"]["total"],
                "unk_tokens": meta["unk_tokens"],
                "oov_rate_pct": meta["oov_rate_pct"],
            }
            for dom, meta in domain_breakdown.items()
        },
        "recomputed_checksums": recomputed_checksums,
    }


def generate_markdown_report_tables(audit_results: Dict[str, Any]) -> str:
    """Generate Markdown tables directly from verified audit results."""
    lines: List[str] = []

    lines.append("### Reconciled Multi-Domain Pilot Corpus Accounting")
    lines.append("")
    lines.append("| Domain Identifier | Parent Sources (Tr/V/Te) | Documents (Tr/V/Te) | Train Tokens | Val Tokens | Test Tokens | Total Tokens | OOV Rate | Evaluation Strength |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    dom_data = audit_results["domain_summary"]
    total_tr_tok = audit_results["split_counts"]["train"]["tokens"]
    total_val_tok = audit_results["split_counts"]["val"]["tokens"]
    total_te_tok = audit_results["split_counts"]["test"]["tokens"]
    total_tok = audit_results["total_tokens"]

    for dom in sorted(dom_data.keys()):
        d = dom_data[dom]
        p_tr, p_v, p_te = d["parents_tr_val_te"].split("/")
        doc_tr, doc_v, doc_te = d["docs_tr_val_te"].split("/")
        tok_tr, tok_v, tok_te = d["tokens_tr_val_te"].split("/")

        # Three-tier evaluation strength indicator:
        # 1. Statistically Adequate Model Evaluation: >= 20 independent parents in val and test
        # 2. Preliminary Domain Evaluation: >= 5 independent parents in val and test (small sample)
        # 3. Pipeline Integrity Smoke-Test: < 5 independent parents (limited volume)
        if int(p_v) >= 20 and int(p_te) >= 20:
            eval_strength = "Statistically Adequate Model Evaluation"
        elif int(p_v) >= 5 and int(p_te) >= 5:
            eval_strength = f"Preliminary Domain Evaluation ({int(p_te)} test parents, {int(tok_te):,} toks; small sample)"
        else:
            eval_strength = f"Pipeline Integrity Smoke-Test ({int(p_te)} test parent; limited)"

        lines.append(
            f"| **`{dom}`** | {p_tr} / {p_v} / {p_te} ({d['parents_total']}) | "
            f"{doc_tr} / {doc_v} / {doc_te} ({d['docs_total']}) | "
            f"{int(tok_tr):,} | {int(tok_v):,} | {int(tok_te):,} | "
            f"**{d['tokens_total']:,}** | {d['oov_rate_pct']:.2f}% | {eval_strength} |"
        )

    tot_p_tr = audit_results["split_counts"]["train"]["parents"]
    tot_p_v = audit_results["split_counts"]["val"]["parents"]
    tot_p_te = audit_results["split_counts"]["test"]["parents"]
    tot_p = audit_results["total_parent_sources"]

    tot_d_tr = audit_results["split_counts"]["train"]["documents"]
    tot_d_v = audit_results["split_counts"]["val"]["documents"]
    tot_d_te = audit_results["split_counts"]["test"]["documents"]
    tot_d = audit_results["retained_record_count"]

    tot_unk = sum(d["unk_tokens"] for d in dom_data.values())
    overall_oov = round(tot_unk / total_tok * 100, 2)

    lines.append(
        f"| **Total Reconciled Corpus** | **{tot_p_tr} / {tot_p_v} / {tot_p_te} ({tot_p})** | "
        f"**{tot_d_tr} / {tot_d_v} / {tot_d_te} ({tot_d})** | "
        f"**{total_tr_tok:,}** | **{total_val_tok:,}** | **{total_te_tok:,}** | "
        f"**{total_tok:,}** | **{overall_oov:.2f}%** | Multi-Domain Pipeline Verified |"
    )

    lines.append("")
    lines.append("### Binary File Checksums and Offset Table Integrity")
    lines.append("")
    lines.append("| Split Name | Documents | Offsets Count | Tokens | Binary Size (bytes) | Offset Binary Size | SHA-256 Checksum |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for split in ["train", "val", "test"]:
        sc = audit_results["split_counts"][split]
        sha = audit_results["recomputed_checksums"][f"{split}.bin"]
        off_bytes = sc["offsets"] * 8
        lines.append(
            f"| `{split}` | {sc['documents']} | {sc['offsets']} | {sc['tokens']:,} | {sc['bytes']:,} | {off_bytes} | `{sha[:20]}...` |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    pilot_path = Path(__file__).resolve().parent
    res = audit_pilot_artifacts(pilot_path)
    print("Audit Results:")
    print(json.dumps(res, indent=2))
    print("\nGenerated Report Tables:")
    print(generate_markdown_report_tables(res))
