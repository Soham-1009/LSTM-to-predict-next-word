"""Binary serialization and memory-mapped dataset preparation.

Implements requirements from Data Pipeline.md:
- Selects uint16 when vocab_size <= 65535, with automatic fallback to uint32
- Encodes documents with <BOS> and <EOS> boundary markers
- Generates train.bin, val.bin, and test.bin
- Generates document boundary offsets for random-access document loading
- Generates comprehensive metadata.json including file sizes, token counts, and SHA-256 checksums
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from backend.model.tokenizer import Tokenizer


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 checksum of a binary file by streaming chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):  # 1 MB chunk
            h.update(chunk)
    return h.hexdigest()


def get_token_dtype(vocab_size: int) -> Tuple[np.dtype, str]:
    """Select uint16 when vocab_size <= 65535, fallback to uint32 otherwise."""
    if vocab_size <= 65535:
        return np.dtype(np.uint16), "uint16"
    return np.dtype(np.uint32), "uint32"


def serialize_split_to_bin(
    documents: Iterable[str],
    tokenizer: Tokenizer,
    output_bin_path: Path | str,
    output_offset_path: Optional[Path | str] = None,
    dtype: Optional[np.dtype] = None,
) -> Tuple[int, int, str]:
    """Encode documents into contiguous binary file and record document boundary offsets.

    Returns:
        total_tokens: Total token count in the binary
        doc_count: Total number of documents serialized
        file_hash: SHA-256 hash of the generated .bin file
    """
    bin_path = Path(output_bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    if dtype is None:
        dtype, _ = get_token_dtype(len(tokenizer))

    token_offsets: List[int] = [0]
    total_tokens = 0
    doc_count = 0

    with open(bin_path, "wb") as f_bin:
        for doc in documents:
            token_ids = tokenizer.encode(doc, add_bos=True, add_eos=True)
            if not token_ids:
                continue

            arr = np.array(token_ids, dtype=dtype)
            arr.tofile(f_bin)

            total_tokens += len(token_ids)
            token_offsets.append(total_tokens)
            doc_count += 1

    # Save document offsets as uint64
    if output_offset_path is not None:
        offset_path = Path(output_offset_path)
        offset_path.parent.mkdir(parents=True, exist_ok=True)
        offsets_arr = np.array(token_offsets, dtype=np.uint64)
        offsets_arr.tofile(str(offset_path))

    file_hash = compute_file_sha256(bin_path)
    return total_tokens, doc_count, file_hash


def load_memmap_tokens(
    bin_path: Path | str, dtype: np.dtype = np.uint16
) -> np.memmap:
    """Load serialized token binary as a zero-copy memory-mapped array."""
    path = Path(bin_path)
    if not path.exists():
        raise FileNotFoundError(f"Serialized binary not found at: {bin_path}")
    return np.memmap(path, dtype=dtype, mode="r")


def generate_metadata_record(
    dataset_version: str,
    source_info: str,
    vocab_size: int,
    dtype_name: str,
    raw_size_bytes: int,
    raw_document_count: int,
    unique_document_count: int,
    duplicate_document_count: int,
    split_counts: Dict[str, Dict[str, Any]],
    split_ratios: Dict[str, float],
    seed: int,
    special_token_ids: Dict[str, int],
    file_checksums: Dict[str, str],
) -> Dict[str, Any]:
    """Construct structured dataset metadata dictionary according to Data Pipeline.md."""
    return {
        "dataset_version": dataset_version,
        "source_info": source_info,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "token_dtype": dtype_name,
        "vocab_size": vocab_size,
        "special_token_ids": special_token_ids,
        "split_ratios": split_ratios,
        "statistics": {
            "raw_size_bytes": raw_size_bytes,
            "raw_document_count": raw_document_count,
            "unique_document_count": unique_document_count,
            "duplicate_document_count": duplicate_document_count,
            "train": split_counts.get("train", {}),
            "val": split_counts.get("val", {}),
            "test": split_counts.get("test", {}),
            "total_tokens": (
                split_counts.get("train", {}).get("tokens", 0)
                + split_counts.get("val", {}).get("tokens", 0)
                + split_counts.get("test", {}).get("tokens", 0)
            ),
        },
        "file_checksums": file_checksums,
    }


def write_metadata_file(
    metadata: Dict[str, Any], output_path: Path | str
) -> None:
    """Persist metadata dictionary to formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
