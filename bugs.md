# Repository Bug Ledger & Audit Tracking

This document tracks identified bugs, architectural edge-cases, concurrency vulnerabilities, and their verified resolutions across the **LSTM Next-Word Prediction** repository.

---

## 1. High Priority (Must Fix)

### BUG-001: Deterministic Document Splitting Produces Empty Validation Split on Small Datasets
* **Component**: Training Preprocessing ([backend/training/preprocess.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/training/preprocess.py#L194-L203))
* **Severity**: Critical (Pipeline Crash)
* **Status**: `RESOLVED`
* **Root Cause**: `split_documents_deterministic` calculated train, validation, and test slice sizes using fixed ratios (85% / 7.5% / 7.5%). For datasets with fewer than 14 documents (such as `SAMPLE_DOCUMENTS` with 10 documents used during `--sample` runs), `int(10 * 0.075)` evaluated to `0`. This generated an empty `val.bin` file (0 bytes), causing downstream pipeline validation (`validate_pipeline.py`) to fail immediately with an empty file exception.
* **Resolution**: Guaranteed a minimum of 1 document for both validation and test sets when total unique documents $n \ge 3$:
  ```python
  if n >= 3:
      n_val = max(int(n * val_ratio), 1)
      n_test = max(int(n * test_ratio), 1)
      n_train = n - n_val - n_test
  else:
      n_train = int(n * train_ratio)
      n_val = int(n * val_ratio)
  ```

---

### BUG-002: In-Place Parameter Mutation Causes Partial Weight Reads During Concurrent Generation
* **Component**: FastAPI Model State Management ([backend/app/main.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/app/main.py#L108-L135))
* **Severity**: High (Model State Corruption under Concurrent Load)
* **Status**: `RESOLVED`
* **Root Cause**: While `load_checkpoint` acquired `_model_lock`, it mutated model parameters in-place via `self.model.load_state_dict()`. The `/generate` endpoint executes iterative multi-token autoregressive decoding (up to 250 forward passes) using a reference `model_ref = service.model` outside the lock. An incoming checkpoint switch during generation mutated parameter tensors mid-stream, producing mixed-weight activations and corrupt text.
* **Resolution**: Made checkpoint swapping atomic at the object reference level. `load_checkpoint` constructs a new `LSTMLanguageModel` instance, loads weights into it, and atomically swaps `self.model = new_model` under `_model_lock`. Ongoing generation calls continue executing on their isolated model snapshot without interference:
  ```python
  new_model = LSTMLanguageModel(...).to(self.device)
  new_model.load_state_dict(checkpoint["model_state_dict"])
  new_model.eval()

  with self._model_lock:
      self.model = new_model
      self.active_checkpoint_name = canonical_name
      self.checkpoint_loaded = True
  ```

---

## 2. Medium Priority (Should Fix)

### BUG-003: Interrupted Corpus Download Leaves Corrupt Archive That Blocks Retries
* **Component**: Corpus Acquisition ([backend/training/download_corpus.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/training/download_corpus.py#L46-L55))
* **Severity**: Medium (Unrecoverable Network Interruption)
* **Status**: `RESOLVED`
* **Root Cause**: `download_and_extract_wikitext` downloaded the 181 MB compressed archive directly to `tgz_path`. If cancelled or interrupted by network dropouts, a partial `.tgz` remained. On subsequent executions, `not tgz_path.exists()` evaluated to `False`, skipping re-download and crashing during `tarfile.open()` with `ReadError`.
* **Resolution**: Downloaded to a temporary file (`wikitext-103.tmp`) and atomically replaced destination via `tmp_tgz.replace(tgz_path)` only after `urllib.request.urlretrieve` completes successfully.

---

### BUG-004: Hardcoded Archive Member Extraction Paths Crash on Packaging Variations
* **Component**: Corpus Extraction ([backend/training/download_corpus.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/training/download_corpus.py#L62-L72))
* **Severity**: Medium (Archive Portability / Mirror Failure)
* **Status**: `RESOLVED`
* **Root Cause**: The tar extractor searched for hardcoded relative member paths (`wikitext-103/train.csv` and `wikitext-103/test.csv`). Mirror archives or tar packagers using `./wikitext-103/` or alternative directory roots raised unhandled `KeyError` exceptions.
* **Resolution**: Replaced hardcoded lookup with suffix-based dynamic member matching:
  ```python
  for suffix in ["train.csv", "test.csv"]:
      member = next((m for m in tar.getmembers() if m.name.endswith(suffix)), None)
      if member is None:
          continue
  ```

---

### BUG-005: Local Module Shadows Python Standard Library `tokenize`
* **Component**: Training Vocabulary Builder ([backend/training/build_vocab.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/training/build_vocab.py))
* **Severity**: Medium (Environment & Tooling Collision)
* **Status**: `RESOLVED`
* **Root Cause**: The vocabulary extraction module was named `tokenize.py`. When scripts in `backend/training/` were run directly or tools added the folder to `sys.path[0]`, any internal import of Python's standard library `tokenize` loaded the local module instead, causing confusing errors and necessitating hacky `sys.path.remove()` workarounds in test suites.
* **Resolution**: Renamed `tokenize.py` to `build_vocab.py` and updated imports in [pipeline.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/training/pipeline.py#L36) and [test_pipeline.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/tests/test_pipeline.py#L29).

---

### BUG-006: Invalid CORS Configuration With Wildcard Origin and Credentials
* **Component**: FastAPI Server Setup ([backend/app/main.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/app/main.py#L198-L204))
* **Severity**: Medium (Browser Fetch / Security Policy Rejection)
* **Status**: `RESOLVED`
* **Root Cause**: `CORSMiddleware` configured `allow_origins=["*"]` simultaneously with `allow_credentials=True`. Standard browser fetch engines reject credentialed requests targeting wildcard origins as a security violation.
* **Resolution**: Set `allow_credentials=False` because the inference API does not rely on session cookies or Authorization headers.

---

### BUG-007: Checkpoint Name Mismatch Between Frontend Fallback and Backend Resolver
* **Component**: API Routing & Client Contracts ([frontend/lib/api.ts](file:///c:/Users/soham/Desktop/Soham/GenAI/frontend/lib/api.ts#L260), [backend/app/main.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/app/main.py#L88-L107))
* **Severity**: Low to Medium (Client Checkpoint Switch 400 Error)
* **Status**: `RESOLVED`
* **Root Cause**: The backend accepted `"baseline"` or `"adapted"`, whereas the frontend fallback metadata supplied filenames `["best_val.pt", "multidomain_adapted.pt"]`. Submitting the raw filename to `/model/select-checkpoint` raised a 400 Bad Request error.
* **Resolution**: Added aliasing in `resolve_checkpoint_path` supporting both canonical names and disk filenames:
  ```python
  norm_name = checkpoint_name.lower().strip()
  if norm_name in ("baseline", "best_val.pt"):
      canonical_name = "baseline"
  elif norm_name in ("adapted", "multidomain_adapted.pt"):
      canonical_name = "adapted"
  ```

---

## 3. Reliability & File Integrity (Nice to Have)

### BUG-008: Non-Atomic Binary Dataset Serialization
* **Component**: Binary Serialization ([backend/training/serialize.py](file:///c:/Users/soham/Desktop/Soham/GenAI/backend/training/serialize.py#L58-L86))
* **Severity**: Low (Interrupted Serialization Leaves Corrupt `.bin`)
* **Status**: `RESOLVED`
* **Root Cause**: `serialize_split_to_bin` wrote memory-mapped binary tokens and offset tables directly to the destination files. An out-of-disk error or process kill produced incomplete `.bin` files that appeared valid to existence checks.
* **Resolution**: Serialized data into `.tmp` files (`f"{bin_path.name}.tmp"`) and committed the file using atomic `os.replace` only upon complete token processing and offset indexing.

---

## 4. Previously Resolved Edge Cases & Hardening

| ID | Issue Description | Component | Fix Summary |
|---|---|---|---|
| **HARDEN-01** | Prompt whitespace stripping caused token continuation text truncation | `frontend/app/page.tsx` | Preserved prompt whitespace semantics using split-based prefix detection instead of unconditional `trim()`. |
| **HARDEN-02** | Punctuation tokens incorrectly preceded by whitespace upon autocomplete | `frontend/app/page.tsx` | Added regex check `^[\.,;:\?!'’%\)\]\}]` to attach punctuation flush with previous token without space. |
| **HARDEN-03** | Race condition during unmanaged client startup | `backend/app/main.py` | Added thread-safe double-checked locking with `_init_lock` inside `ensure_initialized()`. |
| **HARDEN-04** | Multinomial sampling zero-sum / NaN crash | `backend/app/main.py` | Fallback to greedy argmax if temperature/top-k filtering yields uniform zero sum or non-finite values. |
| **HARDEN-05** | Monospace typography on non-code UI elements | `frontend/` | Standardized on `Public_Sans` for headings/body and reserved monospace strictly for literal token predictions. |

---

## 5. Areas Not Checked / Future Monitoring
* **Distributed Training**: Multi-GPU / Multi-node Kaggle or Slurm DDP training execution was verified locally for CPU/single-GPU; remote multi-node gradient synchronization has not been verified in a cluster environment.
* **Streaming Autocomplete (WebSockets/SSE)**: Currently `/predict` and `/generate` operate over standard JSON HTTP POST; high-throughput streaming may require HTTP streaming chunking or WebSocket connections under high concurrent user counts.
