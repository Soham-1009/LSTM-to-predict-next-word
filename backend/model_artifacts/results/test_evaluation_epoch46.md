# Final Test Evaluation of LSTM Language Model — Epoch 46

## 1. Executive Summary

A complete, reproducible evaluation of the trained **LSTMLanguageModel** from checkpoint `best_val.pt` (Epoch 46) was conducted on the previously untouched **WikiText-103 test split**. The evaluation was performed in inference mode with gradients disabled (`torch.inference_mode()`) using PyTorch on local hardware.

| Split | Cross-Entropy Loss | Perplexity | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Evaluated Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation (Epoch 46 Baseline)** | **3.9470** | **51.78** | **32.12%** | **47.14%** | **53.58%** | 7,953,260 |
| **Test (Untouched WikiText-103)** | **3.9174** | **50.27** | **32.43%** | **47.51%** | **53.97%** | **7,932,190** |
| **Difference (Test − Validation)** | **-0.0296** | **-1.51** | **+0.31%** | **+0.38%** | **+0.39%** | — |

---

## 2. Generalization Gap Analysis

- **Test Cross-Entropy Loss:** `3.9174` vs. Validation `3.9470` ($\Delta = -0.0296$)
- **Test Perplexity:** `50.27` vs. Validation `51.78` ($\Delta = -1.51$)
- **Top-1 Accuracy:** `32.43%` vs. Validation `32.12%` ($\Delta = +0.31\\%$)
- **Top-3 Accuracy:** `47.51%` vs. Validation `47.14%` ($\Delta = +0.38\\%$)
- **Top-5 Accuracy:** `53.97%` vs. Validation `53.58%` ($\Delta = +0.39\\%$)

**Interpretation:**
The test cross-entropy loss of 3.9174 is within 0.0296 of the validation loss (3.9470), corresponding to a perplexity delta of -1.51 (50.27 vs 51.78). Top-1 accuracy shifted by +0.31 percentage points. This demonstrates close generalization between validation and test splits of WikiText-103, confirming that the model has not severely overfit to the validation distribution.

---

## 3. Hardware & Execution Diagnostics

- **Compute Device:** `cuda` (`NVIDIA GeForce RTX 3050 Laptop GPU`)
- **VRAM Total:** `4095.5 MB`
- **VRAM Peak during Eval:** `2209.9 MB`
- **Evaluation Duration:** `144.93 seconds` (`2.42 minutes`)
- **Throughput:** `54,729 tokens/second`
- **Total Batches:** `886` batches (Batch Size: `256`)

---

## 4. Model Architecture & Hyperparameters

- **Architecture:** `LSTMLanguageModel`
- **Vocabulary Size:** `30,000` tokens
- **Context Length ($T$):** `35` tokens
- **Embedding Dimension:** `256`
- **Hidden Dimension:** `512`
- **LSTM Layers:** `2`
- **Dropout Rate:** `0.30`
- **Trainable Parameters:** `26,748,208`
- **Loss Criterion:** `CrossEntropyLoss(ignore_index=0)`
- **Special Token IDs:** `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3`

---

## 5. Dataset & Integrity Verification

- **Source Corpus:** WikiText-103
- **Test Binary Path:** `C:\Users\soham\Desktop\Soham\GenAI\backend\datasets\processed\test.bin`
- **Test Documents:** `2,169`
- **Total Split Tokens in Binary:** `7,971,938` tokens
- **Evaluated Target Tokens:** `7,932,190` tokens (100% of causal sequence targets across full test split)
- **SHA-256 Checksum (`test.bin`):** `1672e8ad656257489652bca6cc8784c9a9995c8685793e82e31a9528f87083c2`
- **SHA-256 Checksum (`test_offsets.bin`):** `6af1da244f004cacd68808768c623e0f4a685d4823b5366e2cb879e7d3cf2462`

---

## 6. Verification Checklist

- [x] Complete test split evaluated (all 886 batches, 7,932,190 valid tokens).
- [x] Test loss (3.9174) and perplexity (50.27) are finite and positive.
- [x] Accuracies are valid and monotonic: Top-1 (32.43%) $\le$ Top-3 (47.51%) $\le$ Top-5 (53.97%).
- [x] Checkpoint weights were not modified or overwritten.
- [x] Model evaluated strictly in `eval()` mode with gradient calculation disabled (`torch.inference_mode()`).
