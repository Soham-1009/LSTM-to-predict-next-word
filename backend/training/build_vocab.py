"""Corpus tokenization and vocabulary generation module.

Strictly follows Data Pipeline.md rules:
- Builds vocabulary EXCLUSIVELY from the training split to prevent evaluation leakage
- Preserves mandatory special token IDs:
    <PAD>: 0
    <UNK>: 1
    <BOS>: 2
    <EOS>: 3
- Configurable minimum token frequency (min_freq)
- Configurable maximum vocabulary size (max_vocab_size)
- Persists vocabulary artifact to vocab.json
"""

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
from backend.model.tokenizer import Tokenizer, SPECIAL_TOKENS


def build_vocabulary_from_training_split(
    train_documents: Iterable[str],
    max_vocab_size: int = 30000,
    min_freq: int = 2,
    output_path: Optional[Path | str] = None,
) -> Tokenizer:
    """Build vocabulary strictly from training split documents.

    Counts token frequencies across train_documents, ranks by frequency descending,
    assigns integer IDs, ensures special tokens remain 0..3, and optionally saves
    the resulting Tokenizer artifact to JSON.
    """
    tokenizer = Tokenizer()
    counter: Counter[str] = Counter()

    print("Counting word frequencies across training split documents...")
    doc_count = 0
    total_token_occurrences = 0

    for doc in train_documents:
        tokens = tokenizer.tokenize(doc)
        counter.update(tokens)
        doc_count += 1
        total_token_occurrences += len(tokens)

    print(
        f"Processed {doc_count} training documents with {total_token_occurrences:,} token occurrences. "
        f"Unique raw tokens observed: {len(counter):,}"
    )

    tokenizer.build_vocab(
        token_frequencies=counter,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
    )

    print(f"Final vocabulary size (including special tokens): {len(tokenizer):,}")

    # Verify special token IDs are locked correctly
    assert tokenizer.pad_id == 0, f"Expected <PAD> == 0, got {tokenizer.pad_id}"
    assert tokenizer.unk_id == 1, f"Expected <UNK> == 1, got {tokenizer.unk_id}"
    assert tokenizer.bos_id == 2, f"Expected <BOS> == 2, got {tokenizer.bos_id}"
    assert tokenizer.eos_id == 3, f"Expected <EOS> == 3, got {tokenizer.eos_id}"

    if output_path is not None:
        save_path = Path(output_path)
        tokenizer.save(save_path)
        print(f"Saved tokenizer vocabulary to: {save_path}")

    return tokenizer
