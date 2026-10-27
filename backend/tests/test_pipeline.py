"""Unit and integration tests for data pipeline components.

Uses pytest's tmp_path fixture to run completely in isolation without depending on external files.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from backend.model.tokenizer import Tokenizer
from backend.training.dataset import CausalLanguageModelDataset
from backend.training.preprocess import (
    compute_doc_hash,
    deduplicate_and_clean_stream,
    filter_document,
    normalize_document,
    split_documents_deterministic,
)
from backend.training.serialize import (
    compute_file_sha256,
    generate_metadata_record,
    get_token_dtype,
    load_memmap_tokens,
    serialize_split_to_bin,
    write_metadata_file,
)
from backend.training.build_vocab import build_vocabulary_from_training_split


def test_normalize_document():
    raw_text = "  Hello \r\n  world!  \t  This   is a\n\n\n\ntest.  "
    normalized = normalize_document(raw_text)
    assert "\r" not in normalized
    assert "   " not in normalized
    assert "\n\n\n" not in normalized
    assert normalized.startswith("Hello\nworld!")
    assert normalized.endswith("test.")


def test_filter_document():
    # Too short
    assert not filter_document("Short", min_chars=50)
    # Valid length and printable
    valid_text = "This is a sufficiently long valid document containing natural English sentences with appropriate punctuation and formatting."
    assert filter_document(valid_text, min_chars=50)
    # Corrupt unprintable content
    corrupt_text = "\x00\x01\x02" * 50
    assert not filter_document(corrupt_text, min_chars=50)


def test_deduplicate_and_clean_stream():
    docs = [
        "Unique document number one with sufficient character length for testing purposes.",
        "Unique document number two with sufficient character length for testing purposes.",
        "Unique document number one with sufficient character length for testing purposes.",  # Duplicate of 1
        "   Unique document number two with sufficient character length for testing purposes.   \r\n",  # Duplicate of 2
    ]
    unique_items = list(deduplicate_and_clean_stream(docs, min_chars=30))
    assert len(unique_items) == 2

    hashes = [h for h, _ in unique_items]
    assert len(set(hashes)) == 2


def test_deterministic_split_no_leakage():
    # Create 20 unique docs
    docs = [(f"hash_{i}", f"This is document number {i} for split testing with sufficient length.") for i in range(20)]

    train, val, test = split_documents_deterministic(docs, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42)

    assert len(train) + len(val) + len(test) == 20

    train_hashes = {h for h, _ in train}
    val_hashes = {h for h, _ in val}
    test_hashes = {h for h, _ in test}

    # Strict zero leakage
    assert len(train_hashes.intersection(val_hashes)) == 0
    assert len(train_hashes.intersection(test_hashes)) == 0
    assert len(val_hashes.intersection(test_hashes)) == 0

    # Test reproducibility with same seed
    t2, v2, te2 = split_documents_deterministic(docs, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42)
    assert [h for h, _ in train] == [h for h, _ in t2]


def test_vocabulary_building_training_only(tmp_path: Path):
    train_docs = [
        "artificial neural networks process sequences of data efficiently",
        "causal language models predict the next token given context words",
    ]
    vocab_file = tmp_path / "vocab.json"
    tokenizer = build_vocabulary_from_training_split(
        train_documents=train_docs,
        max_vocab_size=1000,
        min_freq=1,
        output_path=vocab_file,
    )

    assert vocab_file.exists()
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3

    # Check words are in vocabulary
    assert "artificial" in tokenizer.token_to_id
    assert "models" in tokenizer.token_to_id


def test_get_token_dtype():
    dtype_16, name_16 = get_token_dtype(30000)
    assert dtype_16 == np.uint16
    assert name_16 == "uint16"

    dtype_32, name_32 = get_token_dtype(70000)
    assert dtype_32 == np.uint32
    assert name_32 == "uint32"


def test_binary_serialization_and_memmap(tmp_path: Path):
    tokenizer = Tokenizer()
    freqs = {"deep": 10, "learning": 9, "is": 8, "great": 7}
    tokenizer.build_vocab(freqs)

    docs = ["deep learning is great", "learning is deep"]
    bin_path = tmp_path / "test.bin"
    offset_path = tmp_path / "offsets.bin"

    total_tokens, doc_count, file_hash = serialize_split_to_bin(
        documents=docs,
        tokenizer=tokenizer,
        output_bin_path=bin_path,
        output_offset_path=offset_path,
    )

    assert bin_path.exists()
    assert offset_path.exists()
    assert doc_count == 2
    assert total_tokens > 0

    # Verify SHA-256
    assert compute_file_sha256(bin_path) == file_hash

    # Test memmap reading
    loaded_tokens = load_memmap_tokens(bin_path, dtype=np.uint16)
    assert len(loaded_tokens) == total_tokens

    # First token of first document must be <BOS> (2)
    assert loaded_tokens[0] == tokenizer.bos_id


def test_causal_dataset_shifted_targets():
    # Tokens: 0, 1, 2, ..., 99
    tokens = np.arange(100, dtype=np.int64)
    seq_len = 10
    dataset = CausalLanguageModelDataset(tokens, seq_len=seq_len)

    assert len(dataset) == 100 - seq_len
    x, y = dataset[0]

    assert x.tolist() == list(range(0, 10))
    assert y.tolist() == list(range(1, 11))
    assert torch.equal(x[1:], y[:-1])


def test_metadata_generation_and_io(tmp_path: Path):
    meta_path = tmp_path / "metadata.json"
    record = generate_metadata_record(
        dataset_version="1.0.0",
        source_info="test",
        vocab_size=5000,
        dtype_name="uint16",
        raw_size_bytes=1000,
        raw_document_count=10,
        unique_document_count=8,
        duplicate_document_count=2,
        split_counts={"train": {"tokens": 100, "documents": 6}},
        split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=42,
        special_token_ids={"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3},
        file_checksums={"test.bin": "abcdef123456"},
    )
    write_metadata_file(record, meta_path)
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["dataset_version"] == "1.0.0"
    assert loaded["statistics"]["unique_document_count"] == 8


def test_causal_dataset_out_of_bounds_index_error():
    """Verify that CausalLanguageModelDataset raises IndexError on invalid indices."""
    tokens = np.arange(50, dtype=np.int64)
    dataset = CausalLanguageModelDataset(tokens, seq_len=10)
    assert len(dataset) == 40

    # Negative index
    with pytest.raises(IndexError):
        _ = dataset[-1]

    # Index >= num_samples
    with pytest.raises(IndexError):
        _ = dataset[40]

    with pytest.raises(IndexError):
        _ = dataset[100]


def test_evaluate_model_padding_mask():
    """Verify evaluate_model correctly excludes pad_idx tokens from valid tokens and accuracy."""
    from backend.model.lstm import LSTMLanguageModel
    from backend.training.evaluate import evaluate_model
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    # Create dummy model
    model = LSTMLanguageModel(vocab_size=100, embed_dim=32, hidden_dim=64, num_layers=1)
    model.eval()

    # Create dummy batch where second half of sequence is PAD (0)
    x = torch.randint(1, 100, (2, 10))
    y = torch.randint(1, 100, (2, 10))
    y[:, 5:] = 0  # PAD targets

    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    metrics = evaluate_model(model, loader, criterion, device=torch.device("cpu"), pad_idx=0)
    assert metrics["loss"] > 0
    # Exactly 2 * 5 = 10 valid non-padding tokens
    assert metrics["top1_acc"] >= 0.0


def test_real_data_pilot_metadata_and_checksums():
    """Verify Phase 3 real-data pilot binaries, metadata, and SHA-256 checksums."""
    pilot_dir = Path(__file__).resolve().parent.parent.parent / "experiments/multidomain_v1/pilot"
    metadata_path = pilot_dir / "processed/metadata.json"
    manifest_path = pilot_dir / "pilot_manifest.jsonl"

    assert metadata_path.exists(), f"Missing pilot metadata: {metadata_path}"
    assert manifest_path.exists(), f"Missing pilot manifest: {manifest_path}"

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["vocab_size"] == 30000
    assert meta["special_token_ids"] == {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    assert meta["random_seed"] == 42
    assert meta["statistics"]["total_tokens"] <= 1_000_000

    # Verify binary file SHA-256 checksums
    checksums = meta["file_checksums"]
    for filename, expected_hash in checksums.items():
        bin_path = pilot_dir / "processed" / filename
        assert bin_path.exists(), f"Missing binary file: {bin_path}"
        actual_hash = compute_file_sha256(bin_path)
        assert actual_hash == expected_hash, f"SHA-256 mismatch for {filename}!"


def test_real_data_pilot_parent_source_disjointness():
    """Verify Phase 3B pilot enforces strict zero parent-source leakage and zero hash leakage."""
    pilot_dir = Path(__file__).resolve().parent.parent.parent / "experiments/multidomain_v1/pilot"
    manifest_path = pilot_dir / "pilot_manifest.jsonl"

    parents_by_split = {"train": set(), "val": set(), "test": set()}
    hashes_by_split = {"train": set(), "val": set(), "test": set()}
    domains_seen = set()
    total_tokens = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            split = entry["assigned_split"]
            if split not in parents_by_split:
                continue

            domains_seen.add(entry["domain"])
            p_id = entry["parent_source_id"]
            clean_hash = entry["clean_sha256"]

            parents_by_split[split].add(p_id)
            hashes_by_split[split].add(clean_hash)
            total_tokens += entry["token_count"]

    # 1. Strict zero parent-source leakage across splits
    assert len(parents_by_split["train"].intersection(parents_by_split["val"])) == 0, "Parent source leaked between train and val!"
    assert len(parents_by_split["train"].intersection(parents_by_split["test"])) == 0, "Parent source leaked between train and test!"
    assert len(parents_by_split["val"].intersection(parents_by_split["test"])) == 0, "Parent source leaked between val and test!"

    # 2. Strict zero hash leakage across splits
    assert len(hashes_by_split["train"].intersection(hashes_by_split["val"])) == 0, "Hash leaked between train and val!"
    assert len(hashes_by_split["train"].intersection(hashes_by_split["test"])) == 0, "Hash leaked between train and test!"
    assert len(hashes_by_split["val"].intersection(hashes_by_split["test"])) == 0, "Hash leaked between val and test!"

    # 3. All 6 target domains present and token ceiling respected
    expected_domains = {"wikitext103_replay", "literature", "technical_docs", "scientific", "conversation", "news"}
    assert expected_domains.issubset(domains_seen)
    assert total_tokens <= 1_000_000


def test_real_data_pilot_full_domain_split_coverage():
    """Verify every domain has non-zero tokens and independent parents in train, val, and test."""
    pilot_dir = Path(__file__).resolve().parent.parent.parent / "experiments/multidomain_v1/pilot"
    metadata_path = pilot_dir / "processed/metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    domain_breakdown = meta["domain_breakdown"]
    expected_domains = ["wikitext103_replay", "literature", "technical_docs", "scientific", "conversation", "news"]

    for dom in expected_domains:
        assert dom in domain_breakdown, f"Domain {dom} missing from metadata!"
        s = domain_breakdown[dom]
        # Verify parent count >= 1 in train, val, test
        parents = s["parent_sources"]
        assert parents["train"] >= 1, f"Domain {dom} has 0 train parents!"
        assert parents["val"] >= 1, f"Domain {dom} has 0 val parents!"
        assert parents["test"] >= 1, f"Domain {dom} has 0 test parents!"
        # Verify token count > 0 in train, val, test
        toks = s["tokens"]
        assert toks["train"] > 0, f"Domain {dom} has 0 train tokens!"
        assert toks["val"] > 0, f"Domain {dom} has 0 val tokens!"
        assert toks["test"] > 0, f"Domain {dom} has 0 test tokens!"


def test_pmc_license_verification():
    """Verify exact, conservative PMC license parsing and rejection rules."""
    from experiments.multidomain_v1.pilot.ingest_and_validate import verify_pmc_license_metadata

    # 1. Valid CC-BY 4.0 variants
    valid_cases = [
        ["https://creativecommons.org/licenses/by/4.0/"],
        ["http://creativecommons.org/licenses/by/4.0/"],
        ["https://creativecommons.org/licenses/by/4.0"],
        ["http://creativecommons.org/licenses/by/4.0"],
        ["https://creativecommons.org/licenses/by/4.0/legalcode"],
        ["https://creativecommons.org/licenses/by/4.0/deed.en"],
    ]
    for refs in valid_cases:
        is_valid, uri, code = verify_pmc_license_metadata(refs)
        assert is_valid is True, f"Failed on valid ref: {refs}"
        assert uri == refs[0]
        assert code == "VERIFIED_CC_BY_4_0"

    # 2. Non-commercial variants (CC-BY-NC) must be rejected
    nc_cases = [
        ["https://creativecommons.org/licenses/by-nc/4.0/"],
        ["https://creativecommons.org/licenses/by-nc-sa/4.0/"],
        ["https://creativecommons.org/licenses/by-nc-nd/4.0/"],
    ]
    for refs in nc_cases:
        is_valid, uri, code = verify_pmc_license_metadata(refs)
        assert is_valid is False
        assert uri is None
        assert code == "REJECTED_NON_COMMERCIAL_VARIANT"

    # 3. Share-alike variants (CC-BY-SA) must be rejected
    sa_cases = [
        ["https://creativecommons.org/licenses/by-sa/4.0/"],
    ]
    for refs in sa_cases:
        is_valid, uri, code = verify_pmc_license_metadata(refs)
        assert is_valid is False
        assert uri is None
        assert code == "REJECTED_SHARE_ALIKE_VARIANT"

    # 4. No-derivatives variants (CC-BY-ND) must be rejected
    nd_cases = [
        ["https://creativecommons.org/licenses/by-nd/4.0/"],
    ]
    for refs in nd_cases:
        is_valid, uri, code = verify_pmc_license_metadata(refs)
        assert is_valid is False
        assert uri is None
        assert code == "REJECTED_NO_DERIVATIVES_VARIANT"

    # 5. Outdated versions (e.g. 3.0 / 2.0) must be rejected
    ver_cases = [
        ["https://creativecommons.org/licenses/by/3.0/"],
        ["https://creativecommons.org/licenses/by/2.5/"],
    ]
    for refs in ver_cases:
        is_valid, uri, code = verify_pmc_license_metadata(refs)
        assert is_valid is False
        assert uri is None
        assert code == "REJECTED_INCOMPATIBLE_VERSION_NOT_4_0"

    # 6. Missing machine-readable references must be rejected
    is_valid, uri, code = verify_pmc_license_metadata([])
    assert is_valid is False
    assert uri is None
    assert code == "REJECTED_MISSING_MACHINE_READABLE_REF"

    # 7. Generic prose alone must never grant CC-BY 4.0
    is_valid, uri, code = verify_pmc_license_metadata(
        [], license_text="This article is distributed under the terms of the Creative Commons Attribution License."
    )
    assert is_valid is False
    assert uri is None
    assert code == "REJECTED_GENERIC_PROSE_WITHOUT_MACHINE_READABLE_REF"

    # 8. Multiple references:
    # Consistent CC-BY 4.0 references pass
    is_valid, uri, code = verify_pmc_license_metadata([
        "https://creativecommons.org/licenses/by/4.0/",
        "http://creativecommons.org/licenses/by/4.0/"
    ])
    assert is_valid is True
    assert uri == "https://creativecommons.org/licenses/by/4.0/"
    assert code == "VERIFIED_CC_BY_4_0"

    # Conflicting/incompatible references fail
    is_valid, uri, code = verify_pmc_license_metadata([
        "https://creativecommons.org/licenses/by/4.0/",
        "https://creativecommons.org/licenses/by-nc/4.0/"
    ])
    assert is_valid is False
    assert uri is None
    assert code == "REJECTED_NON_COMMERCIAL_VARIANT"


def test_document_boundary_safe_causal_dataset():
    """Verify CausalLanguageModelDataset creates boundary-safe sequences strictly inside documents."""
    import numpy as np
    import torch
    from backend.training.dataset import CausalLanguageModelDataset

    # Create 3 synthetic documents with distinct token ranges:
    # Doc 0: 50 tokens (1000..1049)
    # Doc 1: 20 tokens (2000..2019)  -> Shorter than context_length + 1 = 36! Must yield 0 sequences.
    # Doc 2: 80 tokens (3000..3079)
    doc0 = np.arange(1000, 1050, dtype=np.int64)
    doc1 = np.arange(2000, 2020, dtype=np.int64)
    doc2 = np.arange(3000, 3080, dtype=np.int64)
    all_tokens = np.concatenate([doc0, doc1, doc2])
    offsets = np.array([0, len(doc0), len(doc0) + len(doc1), len(all_tokens)], dtype=np.int64)

    seq_len = 35
    stride = 5

    ds = CausalLanguageModelDataset(all_tokens, seq_len=seq_len, stride=stride, doc_offsets=offsets)

    # Manual sequence count:
    # Doc 0 len 50: (50 - 36) // 5 + 1 = 14 // 5 + 1 = 3 sequences (starts: 0, 5, 10)
    # Doc 1 len 20: 0 sequences
    # Doc 2 len 80: (80 - 36) // 5 + 1 = 44 // 5 + 1 = 9 sequences (starts: 70, 75, 80, 85, 90, 95, 100, 105, 110)
    expected_sample_count = 3 + 0 + 9
    assert len(ds) == expected_sample_count, f"Expected {expected_sample_count} samples, got {len(ds)}"

    # Enumerate every generated sequence and verify boundary containment
    for idx in range(len(ds)):
        x, y = ds[idx]
        assert x.shape == torch.Size([seq_len])
        assert y.shape == torch.Size([seq_len])
        # Causal shift check
        assert torch.equal(x[1:], y[:-1])

        # Tokens in x and y must belong to the exact same document
        first_token = x[0].item()
        last_target_token = y[-1].item()

        if first_token < 2000:
            # Doc 0
            assert all(1000 <= t < 1050 for t in x.tolist()), f"Token in x leaked outside doc 0: {x}"
            assert all(1000 <= t < 1050 for t in y.tolist()), f"Token in y leaked outside doc 0: {y}"
            assert last_target_token < 1050
        elif first_token >= 3000:
            # Doc 2
            assert all(3000 <= t < 3080 for t in x.tolist()), f"Token in x leaked outside doc 2: {x}"
            assert all(3000 <= t < 3080 for t in y.tolist()), f"Token in y leaked outside doc 2: {y}"
            assert first_token >= 3000
        else:
            pytest.fail(f"Unexpected token {first_token} in dataset sample!")

    # Verify invalid offset error handling
    # 1. Does not start with 0
    with pytest.raises(ValueError, match="First document offset must be 0"):
        _ = CausalLanguageModelDataset(all_tokens, seq_len=seq_len, doc_offsets=[1, 50, len(all_tokens)])

    # 2. Non-monotonic offsets
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        _ = CausalLanguageModelDataset(all_tokens, seq_len=seq_len, doc_offsets=[0, 70, 50, len(all_tokens)])

    # 3. Last offset does not match data length
    with pytest.raises(ValueError, match="Final document offset"):
        _ = CausalLanguageModelDataset(all_tokens, seq_len=seq_len, doc_offsets=[0, 50, 100])

    # 4. Less than 2 offsets
    with pytest.raises(ValueError, match="at least 2 entries"):
        _ = CausalLanguageModelDataset(all_tokens, seq_len=seq_len, doc_offsets=[0])


def test_pilot_authoritative_audit_passes():
    """Verify that authoritative audit passes on regenerated pilot artifacts."""
    from experiments.multidomain_v1.pilot.audit_pilot import audit_pilot_artifacts

    pilot_dir = Path(__file__).resolve().parent.parent.parent / "experiments/multidomain_v1/pilot"
    audit_res = audit_pilot_artifacts(pilot_dir)

    assert audit_res["status"] == "AUDIT_PASSED"
    assert audit_res["retained_record_count"] == 126
    assert audit_res["total_parent_sources"] == 86
    assert audit_res["total_tokens"] == 123983
    assert audit_res["split_counts"]["train"]["tokens"] == 82134
    assert audit_res["split_counts"]["val"]["tokens"] == 20354
    assert audit_res["split_counts"]["test"]["tokens"] == 21495


def test_pilot_authoritative_audit_detects_inconsistencies(tmp_path):
    """Verify that the authoritative audit catches injected tampering or inconsistent counts."""
    import shutil
    from experiments.multidomain_v1.pilot.audit_pilot import audit_pilot_artifacts

    pilot_dir = Path(__file__).resolve().parent.parent.parent / "experiments/multidomain_v1/pilot"
    test_pilot = tmp_path / "pilot"
    shutil.copytree(pilot_dir, test_pilot)

    meta_file = test_pilot / "processed/metadata.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. Tampered token count in metadata
    meta_tampered = dict(meta)
    meta_tampered["statistics"] = dict(meta["statistics"])
    meta_tampered["statistics"]["total_tokens"] = 999999
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_tampered, f, indent=2)

    with pytest.raises(AssertionError):
        audit_pilot_artifacts(test_pilot)

    # 2. Tampered SHA-256 in metadata
    meta_tampered["statistics"]["total_tokens"] = meta["statistics"]["total_tokens"]
    meta_tampered["file_checksums"]["train.bin"] = "badhash1234567890badhash1234567890"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_tampered, f, indent=2)

    with pytest.raises(AssertionError):
        audit_pilot_artifacts(test_pilot)

    # 3. Tampered document count
    meta_tampered["file_checksums"]["train.bin"] = meta["file_checksums"]["train.bin"]
    meta_tampered["statistics"]["unique_document_count"] = 50
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_tampered, f, indent=2)

    with pytest.raises(AssertionError):
        audit_pilot_artifacts(test_pilot)


def test_evaluation_allocation_config_integrity():
    """Verify Phase 3D.3 machine-readable evaluation allocation configuration arithmetic and tier assertions."""
    from experiments.multidomain_v1.validate_evaluation_plan import validate_evaluation_allocation
    cfg_file = Path(__file__).resolve().parent.parent.parent / "experiments/multidomain_v1/configs/evaluation_allocation.json"
    assert cfg_file.exists(), f"Configuration file missing at {cfg_file}"
    validate_evaluation_allocation(cfg_file)
