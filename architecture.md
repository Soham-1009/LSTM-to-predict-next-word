# System Architecture

## 1. High-Level Architecture

The system has five layers:

1. Dataset Layer
2. Training Layer
3. Model Artifact Layer
4. FastAPI Inference Layer
5. Next.js Frontend Layer

```text
Raw Corpus
    |
    v
Data Pipeline
(Cleaning / Dedup / Split / Tokenization)
    |
    v
PyTorch LSTM Training
    |
    v
Model + Tokenizer + Config
    |
    v
FastAPI Inference
    |
   HTTP/JSON
    |
    v
Next.js Frontend
    |
    v
User
```

## 2. Model Architecture

```text
Input Text
    |
    v
Tokenizer
    |
    v
Token IDs
    |
    v
Embedding
    |
    v
LSTM Layer 1
    |
    v
LSTM Layer 2
    |
    v
Dropout
    |
    v
Linear Projection
    |
    v
Vocabulary Logits
    |
    v
Softmax / Sampling
    |
    v
Predicted Token
```

### Final Realized Production Architecture & Parameters

| Component | Dimensions / Shape | Calculation | Trainable Parameters |
| :--- | :--- | :--- | :--- |
| **Embedding Layer** | $(30000, 256)$ | $30000 \times 256$ | 7,680,000 |
| **LSTM Layer 1** | Input 256, Hidden 512 | $4 \times (256 \times 512 + 512 \times 512 + 512 + 512)$ | 1,576,960 |
| **LSTM Layer 2** | Input 512, Hidden 512 | $4 \times (512 \times 512 + 512 \times 512 + 512 + 512)$ | 2,101,248 |
| **Linear Projection** | $(512, 30000)$ | $512 \times 30000 + 30000$ | 15,390,000 |
| **Total Parameters** | — | — | **26,748,208** |

- **Production Checkpoint:** `backend/model_artifacts/checkpoints/best_val.pt` (Epoch 46, Val Loss 3.9470, Val PPL 51.78, Test Loss 3.9174, Test PPL 50.27).
- **Latest Training Checkpoint:** `backend/model_artifacts/checkpoints/latest.pt` (Epoch 50, Val Loss 3.9693).
- **Special Tokens:** `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3`.

## 3. Causal Language Modeling

For:
```text
w1 w2 w3 ... wn
```

the model learns:
```text
P(w2 | w1)
P(w3 | w1,w2)
P(w4 | w1,w2,w3)
...
```

Training inputs and targets are shifted by one token:
```text
Input:  w1 w2 w3 w4
Target: w2 w3 w4 w5
```

## 4. Training Flow

```text
Memory-Mapped Dataset
        |
        v
Sequence Sampler
        |
        v
Mini-Batch
        |
        v
LSTM
        |
        v
Linear Layer
        |
        v
Cross-Entropy
        |
        v
Backpropagation
        |
        v
Optimizer
        |
        v
Checkpoint
```

## 5. Checkpoints

Training should maintain:
- A latest checkpoint for interruption recovery.
- A best-validation checkpoint for model selection.
- A final deployment artifact.

The checkpoint should store model state and, where required for resuming, optimizer/scheduler state, training position, validation metric and reproducibility metadata.

The test set must not determine checkpoint selection.

## 6. Inference Flow

```text
User Prompt
    |
    v
API Validation
    |
    v
Tokenization
    |
    v
Sequence Preparation
    |
    v
LSTM Inference
    |
    v
Logits
    |
    +----> Temperature / Top-K / Top-P
    |
    v
Selected Token(s)
    |
    v
Decoding
    |
    v
JSON Response
```

## 7. API

Conceptual endpoints:
```text
GET  /health
POST /predict
POST /generate
GET  /model-info
```

## 8. Frontend

The Next.js application should contain:
- Next-word prediction
- Interactive autocomplete
- Text generation
- Model information
- How-it-works section
- Prediction/generation history

The frontend must not load the PyTorch model directly.

```text
Browser
   |
   v
Next.js
   |
 HTTP
   |
   v
FastAPI
   |
   v
PyTorch LSTM
```

## 9. Deployment Separation

### Training
Raw dataset -> preprocessing -> training -> validation -> best checkpoint -> final test

### Deployment
Best checkpoint + tokenizer + config -> FastAPI -> Next.js -> user
