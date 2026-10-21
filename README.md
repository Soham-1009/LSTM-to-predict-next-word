# LSTM Next-Word Prediction & Text Generation System

A complete Generative AI project implementing a custom PyTorch LSTM language model for real-time next-word prediction (autocomplete) and controlled autoregressive text generation.

> **Note on Architecture:** This system uses a custom 2-layer Recurrent Neural Network (LSTM) language model trained from scratch on WikiText-103. It is **not** a transformer-based Large Language Model (LLM) and does not rely on external cloud APIs or third-party inference endpoints.

---

## Abstract

This project demonstrates the design, end-to-end training, rigorous held-out evaluation, and deployment of a causal recurrent language model. Utilizing an LSTM architecture with 26.75 million parameters across 2 recurrent layers and a 30,000-token vocabulary, the system learns statistical language distributions to perform:
1. **Interactive Autocomplete:** Rapid next-word prediction computing Top-K ranked probability distributions.
2. **Controlled Autoregressive Generation:** Sequence continuation with temperature scaling, Top-K truncation, and nucleus (Top-P) sampling.
The system is served via a lightweight FastAPI backend connected to an interactive Next.js 16 / React 19 web frontend with live CUDA telemetry.

---

## 1. System Overview & Features

- **Primary Feature — Interactive Autocomplete:** Given an input prompt, the model tokenizes the sequence, runs a forward pass through the causal LSTM, and computes a ranked distribution over candidate next words with confidence scores. Clicking any candidate appends it to the active text.
- **Secondary Feature — Autoregressive Text Generation:** The model recursively generates sequence continuations step-by-step using configurable sampling controls:
  - **Greedy Decoding:** Deterministic argmax selection.
  - **Temperature Scaling (0.0–2.0):** Flattens or sharpens token probabilities.
  - **Top-K Filtering:** Restricts candidate selection to the top K highest-probability tokens.
  - **Nucleus (Top-P) Filtering:** Dynamically restricts sampling to the minimal set of tokens whose cumulative probability exceeds threshold P.
- **Tertiary Feature — Empirical Multi-Domain Evaluation & Dynamic Model Switching:** A dedicated interactive UI tab displays measured per-domain cross-entropy loss, perplexity, Top-1/Top-5 accuracy, OOV rates, and side-by-side prompt generations. Allows instant runtime model hot-swapping between the frozen baseline (`best_val.pt`) and adapted multi-domain weights (`multidomain_adapted.pt`).
- **Real-Time Hardware & Model Telemetry:** The Next.js interface continuously monitors the FastAPI backend, displaying active device status (`CUDA` or `CPU`), vocabulary size (30,000), active checkpoint name, and architecture details.

---

## 2. Model Architecture & Specifications

The network is implemented in PyTorch (`backend/model/lstm.py`):

| Parameter | Specification |
| :--- | :--- |
| **Model Type** | Causal Recurrent Language Model (`LSTMLanguageModel`) |
| **Vocabulary Size (V)** | 30,000 tokens |
| **Context Length (T)** | 35 tokens |
| **Embedding Dimension (D_emb)** | 256 |
| **Hidden Units (D_hid)** | 512 |
| **LSTM Layers** | 2 stacked recurrent layers |
| **Inter-Layer Dropout** | 0.30 |
| **Total Parameters** | **26,748,208** trainable parameters |
| **Loss Criterion** | Cross-Entropy Loss with `ignore_index=0` (`<PAD>`) |
| **Special Tokens** | `<PAD>: 0`, `<UNK>: 1`, `<BOS>: 2`, `<EOS>: 3` |

---

## 3. Training & Checkpoint Baseline

- **Training Corpus:** WikiText-103 (~103 million tokens, 28,475 articles).
- **Training Setup:** PyTorch DistributedDataParallel (DDP) across 2 × NVIDIA Tesla T4 GPUs (16 GB each) on Kaggle with NCCL backend and AdamW optimizer.
- **Freeze Baseline Checkpoint:** `backend/model_artifacts/checkpoints/best_val.pt` (Epoch 46, val loss 3.9470, selected model).
- **Latest Training Checkpoint:** `backend/model_artifacts/checkpoints/latest.pt` (Epoch 50, val loss 3.9693, preserved for continuity).
- **Validation vs Held-Out Test Results:**

| Evaluation Split | Cross-Entropy Loss | Perplexity (PPL) | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Evaluated Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation (Epoch 46 Baseline)** | **3.9470** | **51.78** | **32.12%** | **47.14%** | **53.58%** | 7,953,260 |
| **Held-Out Test (WikiText-103)** | **3.9174** | **50.27** | **32.44%** | **47.50%** | **53.97%** | 7,971,915 |
| **Generalization Delta (Test − Val)** | **-0.0296** | **-1.51** | **+0.32%** | **+0.36%** | **+0.39%** | — |

The evaluation demonstrates strong in-domain generalization with zero data leakage between splits. Checkpoint weights remain strictly frozen.

---

---

## 4. Multi-Domain Generalization Pilot Benchmark & Model Adaptation

To evaluate whether the LSTM generalizes beyond its initial Wikipedia training corpus, the model was evaluated on an empirical **pilot benchmark across 6 functional text domains** using a strictly partitioned held-out pilot test split (**17 documents across 11 independent parent sources**, evaluating **21,210 causal sequence targets**).

### 4.1 Data Leakage Prevention & Audit
- **Parent Source Disjointness:** Training, validation, and testing sets use strictly disjoint author and parent sources (0.0% overlap).
- **Near-Duplicate Check:** Exact-match and whitespace-normalized SHA-256 hash overlap is **0.0%**.
- **Document Boundary Invariant:** Causal sliding window sequences are constructed strictly within each document using contiguous document offset tables; zero sequences cross document boundaries. The sum of evaluated targets per domain exactly equals the multi-domain overall total (2,380 + 2,590 + 770 + 1,925 + 8,960 + 4,585 = 21,210 targets, 606 sequences × 35).
- **Audit Verification:** Verified by `experiments/multidomain_v1/pilot/audit_pilot.py` (11/11 mathematical invariants verified: `AUDIT_PASSED`).

### 4.2 Empirical Pilot Benchmark: Frozen Baseline vs Adapted Model

A controlled domain adaptation experiment was executed (`backend/training/train_generalization.py`), fine-tuning from `best_val.pt` for 5 epochs using AdamW (LR = 1 × 10⁻⁴, weight decay 0.01, gradient clipping 1.0) on the multi-domain training split (90 documents, 82,134 tokens) and selecting checkpoints on the validation split (19 documents, 20,354 tokens), while **preserving the exact 30,000-token vocabulary and architecture**. The held-out test split was kept strictly isolated until evaluation.

Both **token-weighted aggregate** and **equal-weight macro-average** metrics are reported:

| Domain / Aggregation | Partition Type | Docs | Total Tokens | Evaluated Targets | OOV Rate | Baseline Loss | Adapted Loss | Baseline PPL | Adapted PPL | Top-1 (Base → Adapt) | Top-5 (Base → Adapt) | Outcome / Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Token-Weighted Aggregate** | **All 21,210 Targets** | **17** | **21,495** | **21,210** | **4.51%** | **4.1360** | **3.6981** | **62.56** | **40.37** | 37.13% → **40.14%** | 54.45% → **57.11%** | **-35.5% PPL** (dominated by technical docs) |
| **Equal-Weight Macro-Average** | **1/6 Weight per Domain** | **17** | **21,495** | **21,210** | **4.51%** | **4.5929** | **4.3110** | **117.98** | **95.19** | 27.68% → **29.32%** | 47.01% → **48.72%** | **-19.3% Mean PPL** (Geom PPL: 98.78 → 74.52) |
| Conversation | Colloquial Dialog | 3 | 2,455 | 2,380 | 4.85% | 5.1094 | **4.9944** | 165.57 | **147.58** | 17.14% → **17.31%** | 38.11% → **39.75%** | Improved (-18.0 PPL) |
| Literature | Narrative Prose | 1 | 2,613 | 2,590 | 3.10% | 5.0380 | **4.8494** | 154.16 | **127.67** | 17.64% → **19.61%** | 39.88% → **41.20%** | Improved (-26.5 PPL) |
| News | Journalistic Articles | 5 | 853 | 770 | 6.68% | 5.2274 | **4.8054** | 186.30 | **122.17** | 20.13% → **21.69%** | 38.96% → **40.91%** | Improved (-64.1 PPL) |
| Scientific | Biomedical (PMC) | 3 | 1,946 | 1,925 | 5.34% | 4.7833 | **4.6987** | 119.49 | **109.81** | 24.05% → **23.84%** | 44.16% → **44.57%** | PPL improved (-9.7), Top-1 slight regression (-0.21%) |
| Technical Docs | IETF / RFC Standards | 3 | 8,990 | 8,960 | 5.41% | 3.5102 | **2.5963** | 33.46 | **13.41** | 54.82% → **61.29%** | 66.61% → **72.13%** | Large improvement (-20.1 PPL) |
| WikiText-103 Replay | In-Domain Reference | 2 | 4,638 | 4,585 | 2.65% | 3.8893 | **3.9221** | 48.88 | **50.51** | 32.28% → **32.17%** | 54.31% → **53.76%** | **Measured Regression** (+1.63 PPL, -0.11% Top-1) |
| **WikiText-103 Retention Slice** | Held-Out Slice (22 Docs) | **22** | **100,820** | **100,380** | **3.82%** | **3.9302** | **3.9760** | **50.92** | **53.30** | 32.29% → **31.79%** | 54.06% → **53.52%** | **Measured Regression** (+2.38 PPL, -0.50% Top-1) |

### 4.3 Reconciliation of WikiText Evaluation Sets & Loss Metrics

To ensure scientific reproducibility and avoid conflating different evaluation splits, the repository distinguishes three separate WikiText evaluations:

1. **Full WikiText-103 Held-Out Test Split (`test_evaluation_epoch46.json`):**
   - **Scope:** Complete official test partition (`backend/datasets/processed/test.bin`).
   - **Scale:** 2,169 Wikipedia articles, 7,971,938 total tokens, **7,932,190 evaluated sequence targets** (226,634 × 35).
   - **Baseline Metrics:** Loss: **3.9174** | Perplexity: **50.27** | Top-1: **32.43%** | Top-5: **53.97%**.
   - **Role:** Authoritative standalone evaluation of the frozen baseline (`best_val.pt`). Never combined into a before-and-after comparison against the pilot.

2. **WikiText-103 Retention Slice (`generalization_comparison.json`):**
   - **Scope:** First 22 articles of the test split (`backend/datasets/processed/offsets/test_offsets.bin`), evaluating **100,380 target tokens** (2,868 × 35).
   - **Metrics:** Baseline Loss: **3.9302** (PPL 50.92) vs. Adapted Loss: **3.9760** (PPL 53.30), Δ = +0.0458.
   - **Role:** Evaluates retention trade-offs on the original source domain. Rather than claiming "zero forgetting", the +0.0458 loss increase (+2.38 PPL, −0.50% Top-1) is disclosed as an empirical **measured regression** resulting from adaptation toward out-of-domain patterns (stability-plasticity trade-off).

3. **Multi-Domain Pilot `wikitext103_replay`:**
   - **Scope:** Exactly 2 held-out Wikipedia articles included in the 17-document pilot benchmark (4,585 target tokens).
   - **Metrics:** Baseline Loss: **3.8893** (PPL 48.88) vs. Adapted Loss: **3.9221** (PPL 50.51), Δ = +0.0328 (+1.63 PPL, −0.11% Top-1).
   - **Role:** Direct in-domain reference evaluated under identical pilot benchmark conditions alongside the other 5 domains.

### 4.4 Scope of Pilot Benchmark & Limitations of Word-Level Language Modeling

- **Pilot Benchmark Scope:** The 17-document evaluation confirms domain adaptability on the sampled text distributions. It demonstrates that the LSTM can adapt to colloquial, narrative, and technical patterns, but does **not** support a claim of universal, open-ended generalization across all human language.
- **Measured Regressions:** Model fine-tuning is not universally monotonic. In addition to the +0.0458 loss regression on the WikiText retention slice and +0.0328 on `wikitext103_replay`, the Scientific domain exhibited a slight measured regression in Top-1 accuracy (24.05% → 23.84%, −0.21%) despite perplexity improving (119.49 → 109.81).
- **Out-Of-Vocabulary (OOV) Rate:** A fixed 30,000 word-level vocabulary maps unseen words (named entities, technical identifiers, medical terms) to `<unk>`. In News (6.68% OOV) and Technical Docs (5.41% OOV), this places an architectural upper bound on next-word accuracy.
- **Top-1 vs Top-5 Accuracy:** In natural language, multiple candidate continuations are grammatically valid. Top-1 penalizes equally valid synonyms; Top-5 (57.11% on the adapted model) provides a more realistic measure of sequence comprehension.
- **Perplexity Interpretation:** Perplexity (exp(loss)) is strictly vocabulary-dependent (V = 30,000) and cannot be directly compared against byte-level or subword models.
- **Macro vs Token-Weighted Interpretation:** The token-weighted aggregate shows a 35.5% perplexity reduction (62.56 → 40.37) because technical documentation accounts for 42.2% of the pilot test targets. The equal-weight macro-average treats all 6 domains equally and shows a 19.3% perplexity reduction (117.98 → 95.19), confirming genuine cross-domain benefit independent of corpus size imbalance.

### 4.5 Commands to Reproduce Pilot Benchmark

```powershell
# 1. Run Data Leakage and Split Invariant Audit
backend\.venv\Scripts\python.exe experiments/multidomain_v1/pilot/audit_pilot.py

# 2. Benchmark Baseline Checkpoint (best_val.pt)
backend\.venv\Scripts\python.exe backend/training/evaluate_multidomain.py --checkpoint backend/model_artifacts/checkpoints/best_val.pt --label baseline_best_val --output backend/model_artifacts/results/generalization_benchmark_baseline.json

# 3. Run Controlled Domain Adaptation Fine-Tuning
backend\.venv\Scripts\python.exe backend/training/train_generalization.py

# 4. Benchmark Adapted Checkpoint (multidomain_adapted.pt)
backend\.venv\Scripts\python.exe backend/training/evaluate_multidomain.py --checkpoint experiments/multidomain_v1/checkpoints/multidomain_adapted.pt --label adapted_multidomain --output backend/model_artifacts/results/generalization_benchmark_adapted.json

# 5. Generate Combined Comparison Report
backend\.venv\Scripts\python.exe backend/training/generate_comparison_report.py
```

---

## 5. Hardware Requirements

- **Inference GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (or any CUDA-compatible GPU with ≥2 GB VRAM).
- **CPU Fallback:** Fully supported on standard x86_64 / ARM CPUs using PyTorch CPU tensors.
- **Host RAM:** Minimum 4 GB RAM.
- **Disk Space:** ~1.5 GB for project files, virtual environment, and 306 MB model weights.

---

## 6. Local Setup and Installation

### 6.1 Prerequisites
- **Python:** 3.10 to 3.12 (tested on 3.11.9)
- **Node.js:** 18.x or higher (tested on Node.js 22.x) and npm 9+
- **Hardware:** CUDA GPU with ≥2 GB VRAM (recommended for fast inference) or standard CPU (fully supported via PyTorch CPU fallback)

### 6.2 Backend Installation

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies from manifest
pip install -r requirements.txt
```

### 6.3 Frontend Installation

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies using the package lockfile
npm ci
```
*(If `npm ci` reports lockfile platform differences on non-Windows OS, use `npm install`.)*

---

## 7. Running the Application

### 7.1 Start the FastAPI Backend (Terminal 1)

```bash
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- API server will start at: `http://127.0.0.1:8000`
- Interactive Swagger documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 7.2 Start the Next.js Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```
- Web UI will be live at: `http://localhost:3000`
- Open `http://localhost:3000` in any modern web browser to interact with autocomplete, autoregressive generation, and the multi-domain generalization dashboard.

### 7.3 Troubleshooting & FAQs

- **Port 8000 or 3000 Already in Use:**
  - If port 8000 is occupied, launch backend on port 8001:
    ```bash
    uvicorn app.main:app --host 127.0.0.1 --port 8001
    ```
  - Point the frontend to the new backend port:
    ```bash
    # Windows PowerShell:
    $env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8001"; npm run dev
    # Linux / macOS:
    NEXT_PUBLIC_API_URL="http://127.0.0.1:8001" npm run dev
    ```
- **Checkpoint Verification:**
  - Run `python scripts/verify_checkpoints.py` from the project root to ensure weights load cleanly into memory and test next-word prediction sanity.
- **Inference vs Training Resumption Requirements:**
  - **Inference & Evaluation:** Fully supported out-of-the-box by the primary package (`GenAI_LSTM_Final_Submission.zip`), which bundles `best_val.pt` (Epoch 46) and `multidomain_adapted.pt` (Epoch 5).
  - **Training Resumption (Optional):** Continuing baseline pretraining from Epoch 50 requires `latest.pt` placed in `backend/model_artifacts/checkpoints/` and the preprocessed WikiText-103 binary tokens in `backend/datasets/processed/`.

---

## 8. Automated Testing & Verification Commands

Run backend test suite (**55 tests** covering API, evaluation, generalization, and pipeline):
```bash
pytest backend/tests -v
```

Verify checkpoint weights integrity and inference compatibility:
```bash
python scripts/verify_checkpoints.py
```

Run frontend data contract smoke tests (**18 tests**):
```bash
cd frontend
npm test
```

Run frontend production build:
```bash
cd frontend
npm run build
```

---

## 9. Submission Deliverables & Release Packages

The release artifacts are maintained in the [`release/`](release/) directory:

- **Primary Submission Package:** [`release/GenAI_LSTM_Final_Submission.zip`](release/GenAI_LSTM_Final_Submission.zip)  
  - **Size:** ~378.74 MB (397,133,921 bytes)  
  - **Status:** **Fully Standalone and Runnable Out-of-the-Box**  
  - **Included Weights:** `best_val.pt` (Epoch 46, primary inference) and `multidomain_adapted.pt` (Epoch 5, adaptation inference)  
- **Lightweight Source-Only Package:** [`release/GenAI_LSTM_Final_Submission_source_only.zip`](release/GenAI_LSTM_Final_Submission_source_only.zip)  
  - **Size:** ~0.60 MB (625,177 bytes)  
  - **Purpose:** Code, configs, tests, and documentation inspection without weights.  
- **Authoritative Checksums:** [`release/RELEASE_CHECKSUMS.sha256`](release/RELEASE_CHECKSUMS.sha256)  
- **Authoritative Inventory Manifest:** [`release/RELEASE_MANIFEST.md`](release/RELEASE_MANIFEST.md)  

---

## 10. Inference Performance Benchmark (RTX 3050 GPU)

Inference performance was benchmarked locally on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) using `scripts/benchmark.py` under sequential execution (`batch_size=1`, `torch.inference_mode()`):

| Task / Endpoint | Pure Model Inference | End-to-End HTTP Latency | Throughput |
| :--- | :---: | :---: | :---: |
| **Cold Model Startup & Weight Loading** | 320.38 ms (p50) | — | — |
| **Next-Word Prediction (`/predict`, Top-K=5)** | **2.20 ms (p50)** | **3.45 ms (p50)** | **289.7 req/s** |
| **Sequence Generation (10 tokens)** | 34.57 ms (p50) | 36.24 ms (p50) | 275.9 tokens/s |
| **Sequence Generation (20 tokens)** | 73.01 ms (p50) | 77.15 ms (p50) | 259.2 tokens/s |
| **Sequence Generation (30 tokens)** | 121.63 ms (p50) | 121.21 ms (p50) | 247.5 tokens/s |
| **Sequence Generation (50 tokens)** | 225.33 ms (p50) | 232.95 ms (p50) | 214.6 tokens/s |

---

## 11. REST API Examples

### Next-Word Prediction (`POST /predict`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The quick brown fox", "top_k": 5}'
```

### Autoregressive Text Generation (`POST /generate`)
```bash
curl -X POST "http://127.0.0.1:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The history of", "max_new_tokens": 15, "temperature": 0.8, "top_k": 40, "top_p": 0.9}'
```

### Checkpoint Switching (`POST /model/select-checkpoint`)
```bash
curl -X POST "http://127.0.0.1:8000/model/select-checkpoint" \
     -H "Content-Type: application/json" \
     -d '{"checkpoint_name": "adapted"}'
```

---

## 12. Architectural Limitations

1. **Context Window:** The LSTM model operates with a 35-token context window; sequential hidden state representations saturate over longer horizons.
2. **Word-Level Tokenization:** Fixed 30,000-word vocabulary maps unknown words to `<unk>`. Future work will implement subword BPE tokenization.
3. **Model Scale:** Designed as an educational, transparent, and reproducible custom recurrent language model; it is not intended to substitute for billion-parameter transformer models.


