"""Corpus preprocessing, cleaning, deduplication, and document splitting pipeline.

Implements requirements from Data Pipeline.md:
- Streaming document extraction from raw files
- Safe UTF-8 decoding with replacement handling
- Line ending and whitespace normalization while preserving meaningful punctuation
- Quality filtering (minimum length, non-empty, high printable ratio)
- SHA-256 exact document deduplication
- Deterministic document-level train/validation/test split to prevent data leakage
"""

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Generator, Iterable, Iterator, List, Optional, Set, Tuple


# Regex pattern to collapse consecutive horizontal spaces without destroying single newlines
HORIZONTAL_SPACES_RE = re.compile(r"[ \t\f\v]+")
# Normalize more than two newlines into double newline (paragraph break)
EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_document(text: str) -> str:
    """Normalize raw document text into clean, standardized form.

    - Replaces Windows/Mac carriage returns with standard LF
    - Collapses horizontal spaces
    - Limits consecutive newlines to paragraph breaks
    - Preserves meaningful punctuation and sentence structures
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace strange Unicode spaces with standard space
    text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]", " ", text)

    # Strip leading/trailing horizontal whitespace per line and collapse internal horizontal spaces
    lines = [HORIZONTAL_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Limit consecutive newlines to paragraph breaks (\n\n)
    text = EXCESSIVE_NEWLINES_RE.sub("\n\n", text)

    return text.strip()


def compute_doc_hash(text: str) -> str:
    """Compute deterministic SHA-256 fingerprint for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def filter_document(
    text: str,
    min_chars: int = 100,
    max_chars: int = 5_000_000,
    min_printable_ratio: float = 0.95,
) -> bool:
    """Validate document against quality filters.

    Returns True if document passes all quality checks, False otherwise.
    """
    if not text:
        return False

    length = len(text)
    if length < min_chars or length > max_chars:
        return False

    # Check printable character ratio to discard corrupt binary fragments
    printable_count = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    if (printable_count / length) < min_printable_ratio:
        return False

    return True


def stream_documents_from_file(file_path: Path | str) -> Generator[str, None, None]:
    """Stream raw documents from a file based on extension (.txt, .jsonl).

    - For .txt files: documents separated by blank lines or whole file if short
    - For .jsonl files: one document per line under "text" or "content" key
    """
    path = Path(file_path)
    if not path.exists():
        return

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text") or record.get("content") or ""
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue
    else:
        # Stream text file line by line without loading entire large corpus into RAM
        buffer: List[str] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "<|endoftext|>" in line:
                    parts = line.split("<|endoftext|>")
                    for i, part in enumerate(parts):
                        buffer.append(part)
                        if i < len(parts) - 1:
                            doc = "".join(buffer).strip()
                            if doc:
                                yield doc
                            buffer = []
                else:
                    buffer.append(line)

        if buffer:
            remaining = "".join(buffer).strip()
            if "\n\n\n" in remaining:
                for doc in remaining.split("\n\n\n"):
                    if doc.strip():
                        yield doc.strip()
            elif remaining:
                yield remaining


def stream_documents_from_directory(
    dir_path: Path | str,
    extensions: Tuple[str, ...] = (".txt", ".jsonl", ".tokens"),
    exclude_sample: bool = True,
) -> Generator[str, None, None]:
    """Iterate through all matching raw files in directory and yield documents.

    Ignores sample_corpus.txt by default so test samples are not mixed into production data.
    """
    directory = Path(dir_path)
    if not directory.exists():
        return

    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            if exclude_sample and file_path.name == "sample_corpus.txt":
                continue
            yield from stream_documents_from_file(file_path)


def deduplicate_and_clean_stream(
    raw_documents: Iterable[str],
    min_chars: int = 100,
    max_chars: int = 5_000_000,
) -> Generator[Tuple[str, str], None, None]:
    """Clean documents and yield unique (sha256_hash, cleaned_text) pairs.

    Keeps track of seen SHA-256 hashes to guarantee exact deduplication.
    """
    seen_hashes: Set[str] = set()

    for raw_doc in raw_documents:
        cleaned = normalize_document(raw_doc)
        if not filter_document(cleaned, min_chars=min_chars, max_chars=max_chars):
            continue

        doc_hash = compute_doc_hash(cleaned)
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            yield doc_hash, cleaned


def split_documents_deterministic(
    documents: List[Tuple[str, str]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.075,
    test_ratio: float = 0.075,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split unique documents deterministically into train, validation, and test sets.

    Document-level splitting prevents sliding-window sequence leakage between splits.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-4, f"Split ratios must sum to 1.0 (got {total_ratio})"

    # Shuffle deterministically with fixed seed
    shuffled = list(documents)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_docs = shuffled[:n_train]
    val_docs = shuffled[n_train : n_train + n_val]
    test_docs = shuffled[n_train + n_val :]

    # In case of small datasets, guarantee at least 1 document per split if total >= 3
    if n >= 3:
        if len(val_docs) == 0 and len(train_docs) > 1:
            val_docs = [train_docs.pop()]
        if len(test_docs) == 0 and len(train_docs) > 1:
            test_docs = [train_docs.pop()]

    return train_docs, val_docs, test_docs
