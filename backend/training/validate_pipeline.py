"""Data Pipeline Integrity Validation Script.

Executes the 7 mandatory integrity checks from Data Pipeline.md:
1. Binary Existence & Non-Empty Check:
   - train.bin, val.bin, test.bin, offsets, metadata.json, and vocab.json must exist and be > 0 bytes.
2. Token ID Range Validation:
   - Every token ID in the serialized binaries must satisfy: 0 <= token_id < vocab_size.
3. Special Token Validation:
   - Mandatory special tokens <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3 exist in vocab.json.
4. Metadata Count Validation:
   - Actual token and document counts in binary files match metadata.json exactly.
5. Checksum Validation:
   - Re-computed SHA-256 hashes of all binary files match the hashes recorded in metadata.json.
6. Zero Document-Hash Leakage:
   - Zero hash intersection between train, val, and test splits (0.0% leakage).
7. PyTorch Causal Language Model Dataset Validation:
   - CausalLanguageModelDataset correctly memory-maps train.bin and yields valid (X, Y) shifted
     pairs of length 35 where Y[t] == X[t+1].

Usage:
    python -m backend.training.validate_pipeline
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import torch

from backend.model.tokenizer import Tokenizer
from backend.training.dataset import CausalLanguageModelDataset
from backend.training.serialize import compute_file_sha256, load_memmap_tokens


def validate_pipeline(
    processed_dir: Path | str = "backend/datasets/processed",
    vocab_path: Path | str = "backend/model_artifacts/tokenizer/vocab.json",
    context_length: int = 35,
) -> bool:
    """Run all 7 mandatory pipeline integrity checks.

    Returns True if all checks pass, raises AssertionError / returns False otherwise.
    """
    proc_path = Path(processed_dir)
    vocab_file = Path(vocab_path)
    metadata_file = proc_path / "metadata.json"

    print("=" * 70)
    print("STARTING DATA PIPELINE INTEGRITY VALIDATION")
    print("=" * 70)

    passed_checks = 0
    total_checks = 7

    # =========================================================================
    # CHECK 1: Binary Existence and Non-Empty Checks
    # =========================================================================
    print("\n[CHECK 1/7] Validating file existence and non-empty files...")
    required_files = [
        proc_path / "train.bin",
        proc_path / "val.bin",
        proc_path / "test.bin",
        proc_path / "offsets" / "train_offsets.bin",
        proc_path / "offsets" / "val_offsets.bin",
        proc_path / "offsets" / "test_offsets.bin",
        metadata_file,
        vocab_file,
    ]

    for req in required_files:
        assert req.exists(), f"Required file missing: {req}"
        size = req.stat().st_size
        assert size > 0, f"File is unexpectedly empty (0 bytes): {req}"
        print(f"  [PASS] {req.name} exists ({size:,} bytes)")

    passed_checks += 1
    print("  -> CHECK 1 PASSED: All 8 required files exist and are non-empty.")

    # Load metadata and tokenizer
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tokenizer = Tokenizer.load(vocab_file)
    vocab_size = len(tokenizer)
    dtype_str = metadata.get("token_dtype", "uint16")
    dtype = np.uint16 if dtype_str == "uint16" else np.uint32

    # =========================================================================
    # CHECK 2: Token ID Range Validation
    # =========================================================================
    print(f"\n[CHECK 2/7] Validating Token ID range in binary files (vocab_size={vocab_size})...")
    for split in ["train", "val", "test"]:
        bin_file = proc_path / f"{split}.bin"
        tokens = load_memmap_tokens(bin_file, dtype=dtype)
        min_id = int(tokens.min())
        max_id = int(tokens.max())
        assert min_id >= 0, f"Negative token ID {min_id} found in {split}.bin!"
        assert max_id < vocab_size, f"Token ID {max_id} >= vocab_size ({vocab_size}) in {split}.bin!"
        print(f"  [PASS] {split}.bin: {len(tokens):,} tokens, IDs range in [{min_id}, {max_id}] strictly < {vocab_size}")

    passed_checks += 1
    print("  -> CHECK 2 PASSED: All token IDs are strictly within valid vocabulary range.")

    # =========================================================================
    # CHECK 3: Special Token Validation
    # =========================================================================
    print("\n[CHECK 3/7] Validating special token IDs (<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3)...")
    assert tokenizer.pad_id == 0, f"Expected <PAD>=0, got {tokenizer.pad_id}"
    assert tokenizer.unk_id == 1, f"Expected <UNK>=1, got {tokenizer.unk_id}"
    assert tokenizer.bos_id == 2, f"Expected <BOS>=2, got {tokenizer.bos_id}"
    assert tokenizer.eos_id == 3, f"Expected <EOS>=3, got {tokenizer.eos_id}"

    # Verify ID-to-token maps back to expected strings
    assert tokenizer.id_to_token[0] == "<PAD>"
    assert tokenizer.id_to_token[1] == "<UNK>"
    assert tokenizer.id_to_token[2] == "<BOS>"
    assert tokenizer.id_to_token[3] == "<EOS>"
    print("  [PASS] Special token IDs locked: <PAD>: 0, <UNK>: 1, <BOS>: 2, <EOS>: 3")

    passed_checks += 1
    print("  -> CHECK 3 PASSED: Special token mapping is correct and verified.")

    # =========================================================================
    # CHECK 4: Metadata Count Validation
    # =========================================================================
    print("\n[CHECK 4/7] Validating token counts against metadata.json...")
    stats = metadata["statistics"]
    for split in ["train", "val", "test"]:
        bin_file = proc_path / f"{split}.bin"
        tokens = load_memmap_tokens(bin_file, dtype=dtype)
        expected_tokens = stats[split]["tokens"]
        actual_tokens = len(tokens)
        assert actual_tokens == expected_tokens, (
            f"Token count mismatch for {split}.bin: metadata says {expected_tokens}, file has {actual_tokens}"
        )

        offset_file = proc_path / "offsets" / f"{split}_offsets.bin"
        offsets = np.memmap(offset_file, dtype=np.uint64, mode="r")
        expected_docs = stats[split]["documents"]
        actual_docs = len(offsets) - 1
        assert actual_docs == expected_docs, (
            f"Document count mismatch for {split}_offsets.bin: metadata says {expected_docs}, found {actual_docs}"
        )
        print(f"  [PASS] {split}: {actual_tokens:,} tokens, {actual_docs} documents match metadata exactly")

    passed_checks += 1
    print("  -> CHECK 4 PASSED: Metadata counts perfectly match binary data.")

    # =========================================================================
    # CHECK 5: Checksum Validation
    # =========================================================================
    print("\n[CHECK 5/7] Validating SHA-256 checksums of binary and offset files...")
    recorded_checksums = metadata.get("file_checksums", {})
    assert len(recorded_checksums) > 0, "No file_checksums recorded in metadata.json!"

    for filename, expected_hash in recorded_checksums.items():
        # Find file in proc_path or offsets/
        target_path = proc_path / filename if (proc_path / filename).exists() else proc_path / "offsets" / filename
        assert target_path.exists(), f"File {filename} missing for checksum verification!"

        actual_hash = compute_file_sha256(target_path)
        assert actual_hash == expected_hash, (
            f"SHA-256 checksum mismatch for {filename}!\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )
        print(f"  [PASS] {filename}: SHA-256 verified ({actual_hash[:16]}...)")

    passed_checks += 1
    print("  -> CHECK 5 PASSED: All file SHA-256 checksums match metadata records.")

    # =========================================================================
    # CHECK 6: Zero Document-Hash Leakage
    # =========================================================================
    print("\n[CHECK 6/7] Validating zero document-hash leakage across splits...")
    # Read each document using offsets and verify no duplicate hashes across splits
    def get_document_hashes(split_name: str) -> Set[str]:
        bin_file = proc_path / f"{split_name}.bin"
        offset_file = proc_path / "offsets" / f"{split_name}_offsets.bin"
        tokens = load_memmap_tokens(bin_file, dtype=dtype)
        offsets = np.memmap(offset_file, dtype=np.uint64, mode="r")
        hashes = set()
        for i in range(len(offsets) - 1):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            doc_tokens = tokens[start:end]
            # Compute hash of document token sequence
            doc_hash = compute_file_sha256_of_bytes(doc_tokens.tobytes())
            hashes.add(doc_hash)
        return hashes

    def compute_file_sha256_of_bytes(b: bytes) -> str:
        import hashlib
        return hashlib.sha256(b).hexdigest()

    train_hashes = get_document_hashes("train")
    val_hashes = get_document_hashes("val")
    test_hashes = get_document_hashes("test")

    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)

    assert len(train_val_overlap) == 0, f"Leakage detected: {len(train_val_overlap)} documents shared between train and val!"
    assert len(train_test_overlap) == 0, f"Leakage detected: {len(train_test_overlap)} documents shared between train and test!"
    assert len(val_test_overlap) == 0, f"Leakage detected: {len(val_test_overlap)} documents shared between val and test!"
    print(
        f"  [PASS] Split overlap: train & val={len(train_val_overlap)}, "
        f"train & test={len(train_test_overlap)}, val & test={len(val_test_overlap)} (Strictly 0.0% leakage)"
    )

    passed_checks += 1
    print("  -> CHECK 6 PASSED: Zero document leakage confirmed across train, val, and test splits.")

    # =========================================================================
    # CHECK 7: PyTorch Causal Dataset Shifted (X, Y) Validation
    # =========================================================================
    print(f"\n[CHECK 7/7] Validating CausalLanguageModelDataset shifted target objective (T={context_length})...")
    train_bin = proc_path / "train.bin"
    train_tokens = load_memmap_tokens(train_bin, dtype=dtype)

    dataset = CausalLanguageModelDataset(train_tokens, seq_len=context_length)
    assert len(dataset) > 0, f"Dataset has 0 samples with seq_len={context_length} (total tokens: {len(train_tokens)})"

    # Sample a batch of items and verify the causal shift: Y[t] == X[t+1]
    for sample_idx in [0, min(5, len(dataset) - 1), len(dataset) - 1]:
        x, y = dataset[sample_idx]
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        assert isinstance(y, torch.Tensor), f"Expected torch.Tensor, got {type(y)}"
        assert x.shape == (context_length,), f"Expected x shape ({context_length},), got {x.shape}"
        assert y.shape == (context_length,), f"Expected y shape ({context_length},), got {y.shape}"

        # Verify causal shifted target: x[1:] must equal y[:-1]
        assert torch.equal(x[1:], y[:-1]), (
            f"Causal shift violation at index {sample_idx}! x[1:] != y[:-1]\n"
            f"x: {x.tolist()}\ny: {y.tolist()}"
        )

    print(f"  [PASS] Dataset contains {len(dataset):,} valid training sequences.")
    print(f"  [PASS] Verified shifted objective: Y[t] == X[t+1] for context length {context_length}.")

    passed_checks += 1
    print("  -> CHECK 7 PASSED: CausalLanguageModelDataset loads memmap and yields correct causal targets.")

    print("\n" + "=" * 70)
    print(f"ALL {passed_checks}/{total_checks} INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


def main():
    try:
        success = validate_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
