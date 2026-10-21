# Dataset Specification

## 1. Purpose

Provide English text for training the project's LSTM causal language model.

The model learns the probability of the next token from preceding context.

## 2. Target Size

Initial target:
**Approximately 1 GB of clean English text**

Final processed size must be measured from the actual artifacts.

## 3. Source Requirements

Use legally usable English-language text.

Potential source categories:
- Public-domain books
- Stories
- Articles
- Other appropriately licensed English text

The exact final source list should be recorded in the final project documentation.

## 4. Content Quality

Documents should be:
- Non-empty
- Predominantly English
- Text-readable
- Free of obvious corruption
- Long enough to provide useful context
- Deduplicated

## 5. Preprocessing

The dataset pipeline should perform:
1. Safe decoding
2. Unicode normalization where appropriate
3. Whitespace normalization
4. Invalid-document filtering
5. Empty-document removal
6. Minimum-length filtering
7. Exact duplicate removal
8. Document-level splitting
9. Tokenization
10. Integer encoding

Meaningful punctuation should be preserved unless a documented experiment changes the rule.

## 6. Tokenizer

Required special tokens:
```text
<PAD>
<UNK>
<BOS>
<EOS>
```

Their IDs must remain consistent across training, evaluation and inference.

The tokenizer artifact should store:
- token-to-ID mapping
- ID-to-token mapping
- vocabulary size
- special-token IDs
- tokenization configuration

## 7. Vocabulary

Initial target:
**Approximately 30,000–50,000 tokens**

The final vocabulary is determined from the actual processed corpus.

Unknown tokens should map to:
```text
<UNK>
```

## 8. Dataset Split

Split at document level.

Initial target range:
```text
Training:   85–90%
Validation: 5–10%
Test:       5–10%
```

The exact final ratio must be recorded in metadata.

## 9. Why Document-Level Splitting?

A document can produce many overlapping sequences:

```text
A B C D E
B C D E F
C D E F G
```

Randomly splitting these sequences can place very similar examples in both training and test data.

Document-level splitting reduces this leakage risk.

## 10. Causal Training Sequences

For context length `T`:
```text
X = tokens[i : i + T]
Y = tokens[i + 1 : i + T + 1]
```

Example:
```text
Input:  once upon a time the
Target: upon a time the little
```

## 11. Document Boundary Policy

Training sequences should respect document boundaries so unrelated documents are not unintentionally joined.

## 12. Storage

A memory-mapped integer representation is preferred for a large corpus.

Conceptual layout:
```text
datasets/
├── raw/
│   └── source files
└── processed/
    ├── train.bin
    ├── val.bin
    ├── test.bin
    ├── offsets/
    └── metadata.json
```

Tokenizer data can be stored separately.

## 13. Metadata

Final metadata should record:
```text
Dataset version
Source information
Raw size
Processed size
Document count
Training token count
Validation token count
Test token count
Vocabulary size
Special-token IDs
Split ratios
Random seed
Processing configuration
Processing timestamp
```

## 14. Validation

Before training, verify:
- Required files exist.
- Token IDs are valid.
- Vocabulary IDs are valid.
- Special tokens exist.
- Metadata matches actual data.
- Offsets are valid.
- Split boundaries are valid.
- Duplicate documents do not cross splits.
- Dataset loaders can read all splits.
- Sequence generation respects document boundaries.

## 15. Leakage Policy

Validation may be used for:
- Model selection
- Hyperparameter tuning
- Early stopping
- Checkpoint selection

The test set must remain untouched until final held-out evaluation.

## 16. Reproducibility

Maintain enough metadata to reproduce:

```text
Raw Data
   |
   v
Cleaning
   |
   v
Deduplication
   |
   v
Document Split
   |
   v
Tokenizer
   |
   v
Integer Encoding
   |
   v
Serialized Dataset
```

## 17. Final Dataset Record

Populate after the dataset has actually been prepared (measured values from `backend/datasets/processed/metadata.json`):

```text
Raw dataset size:          542,479,871 bytes (~517.3 MB)
Processed dataset size:    210,156,112 bytes (train: 178.3 MB, val: 15.9 MB, test: 15.9 MB)
Unique documents:          28,906 (29,595 raw documents, 689 duplicate documents removed)
Training tokens:           89,152,843 tokens (24,570 documents, 85%)
Validation tokens:         7,953,275 tokens (2,167 documents, 7.5%)
Test tokens:               7,971,938 tokens (2,169 documents, 7.5%)
Total tokens:              105,078,056 tokens
Vocabulary size:           30,000 tokens
Token encoding:            uint16
Sequence context length:   35 tokens
Dataset version:           1.0.0
Source list:               WikiText-103 (verified Good & Featured English Wikipedia articles)
Processing date:           2026-09-21T13:28:06 UTC
```

All values above are exact measurements recorded in `backend/datasets/processed/metadata.json`.
