"""End-to-end Data Pipeline Orchestrator.

Orchestrates the complete dataset lifecycle:
1. Ingests raw text corpus (or generates sample dataset)
2. Performs cleaning and text normalization
3. Executes exact SHA-256 deduplication
4. Computes deterministic document-level split (85% / 7.5% / 7.5%)
5. Builds vocabulary strictly from the training split
6. Encodes documents into integer token IDs with document boundary markers
7. Serializes to train.bin, val.bin, test.bin and boundary offsets
8. Computes SHA-256 checksums and generates metadata.json

Usage:
    python -m backend.training.pipeline --sample
    python -m backend.training.pipeline --raw-dir backend/datasets/raw --out-dir backend/datasets/processed
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from backend.training.preprocess import (
    deduplicate_and_clean_stream,
    split_documents_deterministic,
    stream_documents_from_directory,
    stream_documents_from_file,
)
from backend.training.serialize import (
    compute_file_sha256,
    generate_metadata_record,
    get_token_dtype,
    serialize_split_to_bin,
    write_metadata_file,
)
from backend.training.tokenize import build_vocabulary_from_training_split


SAMPLE_DOCUMENTS = [
    # Document 1: Neural Networks
    "An artificial neural network is an interconnected group of nodes, inspired by a simplification of neurons in a brain. Each node represents an artificial neuron which can process inputs and forward signals to other nodes. Neurons compute an activation function applied to a weighted sum of inputs.",
    # Document 2: Language Modeling
    "A statistical language model is a probability distribution over sequences of words. Given such a sequence of length m, it assigns a probability to the whole sequence. Causal language models compute the probability of the next word given all previous context words.",
    # Document 3: Recurrent Neural Networks
    "A recurrent neural network is a class of artificial neural networks where connections between nodes can create a cycle, allowing output from some nodes to affect subsequent input to the same nodes. This allows it to exhibit temporal dynamic behavior.",
    # Document 4: LSTM Architectures
    "Long short-term memory is an artificial neural network architecture used in the fields of artificial intelligence and deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections, enabling it to process entire sequences of data.",
    # Document 5: Natural Language Processing
    "Natural language processing is an interdisciplinary subfield of computer science and linguistics. It is primarily concerned with giving computers the ability to support and manipulate human language.",
    # Document 6: Generative Artificial Intelligence
    "Generative artificial intelligence is artificial intelligence capable of generating text, images, videos, or other data using generative models, often in response to prompts. Generative AI models learn the patterns and structure of their input training data.",
    # Document 7: Deep Learning Optimization
    "Optimization in deep learning refers to the process of adjusting model parameters to minimize a loss function. Stochastic gradient descent and its variants like Adam and AdamW are commonly employed to update model weights efficiently across mini-batches.",
    # Document 8: Machine Learning Evaluation
    "Evaluation of causal language models relies on metrics like cross-entropy loss, perplexity, and Top-K accuracy. Lower perplexity indicates that the model is more confident and accurate in its next-word predictions on held-out test data.",
    # Document 9: DUPLICATE of Document 1 (Tests SHA-256 deduplication!)
    "An artificial neural network is an interconnected group of nodes, inspired by a simplification of neurons in a brain. Each node represents an artificial neuron which can process inputs and forward signals to other nodes. Neurons compute an activation function applied to a weighted sum of inputs.",
    # Document 10: DUPLICATE of Document 4 (Tests SHA-256 deduplication!)
    "Long short-term memory is an artificial neural network architecture used in the fields of artificial intelligence and deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections, enabling it to process entire sequences of data.",
    # Document 11: Text Tokenization
    "Tokenization is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. The list of tokens becomes input for further processing such as parsing or text mining.",
    # Document 12: Autoregressive Generation
    "Autoregressive generation generates tokens one at a time, appending each sampled token back to the input context. Sampling techniques such as greedy decoding, temperature scaling, and nucleus top-p sampling balance coherence and diversity.",
]


def run_pipeline(
    raw_dir: Path | str,
    output_dir: Path | str,
    vocab_artifact_path: Path | str,
    max_vocab_size: int = 30000,
    min_freq: int = 2,
    train_ratio: float = 0.85,
    val_ratio: float = 0.075,
    test_ratio: float = 0.075,
    seed: int = 42,
    sample_mode: bool = False,
) -> None:
    """Execute complete dataset ingestion, cleaning, deduplication, splitting, and serialization."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING DATA PIPELINE INGESTION & PROCESSING")
    print("=" * 70)

    # 1. Document Extraction
    raw_docs: List[str] = []
    total_raw_bytes = 0

    if sample_mode:
        print("Running in SAMPLE MODE: creating and ingesting sample documents...")
        sample_file = raw_path / "sample_corpus.txt"
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("\n\n<|endoftext|>\n\n".join(SAMPLE_DOCUMENTS))
        raw_docs = list(SAMPLE_DOCUMENTS)
        total_raw_bytes = sum(len(d.encode("utf-8")) for d in raw_docs)
    else:
        print(f"Streaming raw documents from: {raw_path}")
        for file in raw_path.glob("*"):
            if file.is_file() and file.suffix.lower() in {".txt", ".jsonl"}:
                total_raw_bytes += file.stat().st_size
        raw_docs = list(stream_documents_from_directory(raw_path))

    raw_doc_count = len(raw_docs)
    print(f"Ingested {raw_doc_count} raw documents (Total raw size: {total_raw_bytes:,} bytes).")

    if raw_doc_count == 0:
        print(f"Error: No valid documents found in {raw_path}.")
        sys.exit(1)

    # 2. Cleaning & 3. SHA-256 Deduplication
    print("\nApplying cleaning, whitespace normalization, and SHA-256 deduplication...")
    cleaned_unique_docs: List[Tuple[str, str]] = list(
        deduplicate_and_clean_stream(raw_docs, min_chars=50 if sample_mode else 100)
    )
    unique_count = len(cleaned_unique_docs)
    duplicates_removed = raw_doc_count - unique_count
    print(
        f"Deduplication complete: {unique_count} unique valid documents retained. "
        f"({duplicates_removed} duplicates/invalid filtered out)."
    )

    # 4. Document-Level Splitting
    print(f"\nSplitting documents ({train_ratio*100:.1f}% train / {val_ratio*100:.1f}% val / {test_ratio*100:.1f}% test)...")
    train_docs, val_docs, test_docs = split_documents_deterministic(
        cleaned_unique_docs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    print(f"Document split: Train={len(train_docs)}, Val={len(val_docs)}, Test={len(test_docs)}")

    # 5. Vocabulary Construction (Training Split Only)
    print("\nBuilding vocabulary exclusively from training split...")
    # In sample mode, set min_freq=1 to retain sample tokens
    effective_min_freq = 1 if sample_mode else min_freq
    tokenizer = build_vocabulary_from_training_split(
        train_documents=(doc for _, doc in train_docs),
        max_vocab_size=max_vocab_size,
        min_freq=effective_min_freq,
        output_path=vocab_artifact_path,
    )

    # 6. Binary Serialization & Offset Generation
    vocab_size = len(tokenizer)
    dtype, dtype_name = get_token_dtype(vocab_size)
    print(f"\nSerializing tokenized splits to binary using dtype '{dtype_name}' (vocab_size={vocab_size})...")

    offsets_dir = out_path / "offsets"
    offsets_dir.mkdir(parents=True, exist_ok=True)

    splits_data = [
        ("train", train_docs, out_path / "train.bin", offsets_dir / "train_offsets.bin"),
        ("val", val_docs, out_path / "val.bin", offsets_dir / "val_offsets.bin"),
        ("test", test_docs, out_path / "test.bin", offsets_dir / "test_offsets.bin"),
    ]

    split_counts = {}
    file_checksums = {}

    for split_name, docs, bin_file, offset_file in splits_data:
        tokens, d_count, b_hash = serialize_split_to_bin(
            documents=(d for _, d in docs),
            tokenizer=tokenizer,
            output_bin_path=bin_file,
            output_offset_path=offset_file,
            dtype=dtype,
        )
        offset_hash = compute_file_sha256(offset_file)
        split_counts[split_name] = {
            "documents": d_count,
            "tokens": tokens,
            "bytes": bin_file.stat().st_size,
        }
        file_checksums[bin_file.name] = b_hash
        file_checksums[offset_file.name] = offset_hash
        print(f" -> {split_name}.bin: {tokens:,} tokens ({d_count} docs, {bin_file.stat().st_size:,} bytes)")

    # 7. Metadata Generation
    metadata_path = out_path / "metadata.json"
    metadata = generate_metadata_record(
        dataset_version="1.0.0",
        source_info="sample_corpus" if sample_mode else "english_corpus",
        vocab_size=vocab_size,
        dtype_name=dtype_name,
        raw_size_bytes=total_raw_bytes,
        raw_document_count=raw_doc_count,
        unique_document_count=unique_count,
        duplicate_document_count=duplicates_removed,
        split_counts=split_counts,
        split_ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
        seed=seed,
        special_token_ids={
            "<PAD>": tokenizer.pad_id,
            "<UNK>": tokenizer.unk_id,
            "<BOS>": tokenizer.bos_id,
            "<EOS>": tokenizer.eos_id,
        },
        file_checksums=file_checksums,
    )
    write_metadata_file(metadata, metadata_path)
    print(f"\nMetadata record successfully generated and saved to: {metadata_path}")
    print("=" * 70)
    print("DATA PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Data Pipeline Orchestrator for LSTM Language Model")
    parser.add_argument("--raw-dir", type=str, default="backend/datasets/raw", help="Directory containing raw text files")
    parser.add_argument("--out-dir", type=str, default="backend/datasets/processed", help="Directory for serialized binaries")
    parser.add_argument("--vocab-path", type=str, default="backend/model_artifacts/tokenizer/vocab.json", help="Path to save vocab.json")
    parser.add_argument("--vocab-size", type=int, default=30000, help="Maximum vocabulary size")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.075, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.075, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    parser.add_argument("--sample", action="store_true", help="Generate and process sample corpus")

    args = parser.parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.out_dir,
        vocab_artifact_path=args.vocab_path,
        max_vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        sample_mode=args.sample,
    )


if __name__ == "__main__":
    main()
