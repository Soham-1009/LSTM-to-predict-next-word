# Data Pipeline

## 1. Objective

Convert raw English text into clean, validated, tokenized and efficiently serialized data for LSTM training.

## 2. End-to-End Pipeline

```text
Raw Text
   |
   v
Document Extraction
   |
   v
Cleaning / Normalization
   |
   v
Duplicate Detection
   |
   v
Document Validation
   |
   v
Document-Level Split
   |
   v
Tokenizer / Vocabulary
   |
   v
Integer Encoding
   |
   v
Binary Serialization
   |
   v
Integrity Validation
   |
   v
Training Loader
```

## 3. Raw Data & Final Processed Corpus Statistics

- **Raw Corpus:** WikiText-103 English Wikipedia Articles (542,479,871 bytes, 29,595 raw documents).
- **Deduplication:** 689 duplicate documents removed via SHA-256 fingerprinting → 28,906 unique documents.
- **Split Distribution:**
  - **Train (85%):** 24,570 documents, 89,152,843 tokens, 178,305,686 bytes (`train.bin`).
  - **Validation (7.5%):** 2,167 documents, 7,953,275 tokens, 15,906,550 bytes (`val.bin`).
  - **Test (7.5%):** 2,169 documents, 7,971,938 tokens, 15,943,876 bytes (`test.bin`).
- **Total Processed Tokens:** 105,078,056 tokens encoded as `uint16` in memory-mapped binary format.

## 4. Cleaning

The pipeline should:
- Decode text safely.
- Normalize line endings.
- Normalize unnecessary whitespace.
- Remove invalid/corrupt documents.
- Remove empty documents.
- Apply a minimum useful document length.
- Preserve meaningful punctuation.

## 5. Deduplication

Deduplicate normalized documents before the dataset split.

A normalized document may be fingerprinted with SHA-256:

```text
Normalized Document
        |
        v
     SHA-256
        |
        v
Document Fingerprint
```

Identical fingerprints are exact duplicates.

## 6. Document-Level Splitting

Split documents rather than individual sliding windows.

Initial planning range:
```text
Training:   85–90%
Validation: 5–10%
Test:       5–10%
```

The final ratio must be stored in dataset metadata.

This reduces leakage from nearly identical windows originating from the same source document.

## 7. Tokenizer

Required special tokens:
```text
<PAD>
<UNK>
<BOS>
<EOS>
```

The tokenizer artifact should store:
- token-to-ID mapping
- ID-to-token mapping
- vocabulary size
- special-token mapping
- tokenization configuration

## 8. Vocabulary

Initial target: approximately 30,000–50,000 tokens.

The final vocabulary size is determined from the processed corpus.

Out-of-vocabulary tokens map to:
```text
<UNK>
```

## 9. Sequence Construction

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

This is a causal language-model objective.

## 10. Document Boundaries

Sequence creation must respect document boundaries so unrelated documents are not unintentionally connected.

## 11. Serialization

For a large corpus, prefer memory-mapped integer data rather than millions of Python objects in RAM.

Conceptual layout:
```text
datasets/
├── raw/
└── processed/
    ├── train.bin
    ├── val.bin
    ├── test.bin
    ├── offsets/
    └── metadata.json
```

## 12. Validation

Verify:
- Required files exist.
- Token IDs are within vocabulary range.
- Special tokens exist.
- Metadata counts match data.
- Offsets are valid.
- Split boundaries are valid.
- Duplicate leakage is not present.
- All dataset splits can be loaded.
- Sequence construction is correct.

## 13. Reproducibility

Record:
- dataset version
- source identifier
- raw size
- processing rules
- deduplication method
- random seed
- split ratio
- vocabulary size
- special-token IDs
- token counts
- processing timestamp
- file hashes where useful

## 14. Completion Criteria

The pipeline is complete when:
1. Raw data is available.
2. Documents are cleaned.
3. Exact duplicates are removed.
4. Documents are split.
5. Tokenizer is generated.
6. Corpus is encoded.
7. Serialized datasets are created.
8. Integrity validation passes.
9. Metadata is generated.
