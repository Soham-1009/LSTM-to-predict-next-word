# Backend — LSTM Next Word Predictor

FastAPI service that serves predictions from a trained PyTorch LSTM model.

## Quick start

> **Python version:** this project targets **Python 3.10 or 3.11
> (64-bit)**. The pinned `torch==2.3.0` and `numpy==1.26.4` wheels
> **do not exist** for Python 3.12+ — `pip install` will fail with
> "Could not find a version that satisfies the requirement". If you
> see Python 3.13 or 3.14 when you run `python --version`, install
> 3.11 from <https://www.python.org/downloads/windows/> and use
> `py -3.11` or the full path to create the venv.

```powershell
# 1. Create + activate a Python 3.11 venv
cd C:\Users\soham\Desktop\Soham\GenAI\backend
python -m venv .venv                # or: py -3.11 -m venv .venv
.\.venv\Scripts\activate
python --version                    # MUST say 3.10.x or 3.11.x

# 2. Install deps
pip install -r requirements.txt     # CPU
# OR for your 3050 4GB:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt

# 3. Verify
python -c "import torch, fastapi, pydantic, numpy; print('OK', torch.__version__, torch.cuda.is_available())"
```

Once the install is good, jump to **Train the model** below.

## Train the model

Training produces four artifacts under `backend/model/`:

| File | Purpose |
| ---- | ------- |
| `lstm_model.pth` | Trained weights |
| `tokenizer.pkl` | Word-to-id / id-to-word maps |
| `config.json` | Architecture hyperparameters |
| `training_history.json` | Per-epoch metrics + corpus metadata |

### Pick a corpus

`train.py` accepts three corpus sources via `--corpus`:

| `--corpus` | What it does | Use it when |
| ---------- | ------------ | ----------- |
| `bundled`  | Loads the small `sample_corpus.txt` in the repo | You want a 30-second smoke test |
| `gutenberg` | Downloads a public-domain book from Project Gutenberg | You want real predictions for the demo |
| `file`     | Reads any plain-text file at `--corpus-path` | You have your own corpus |

For `--corpus gutenberg`, the available book keys are:

| `--corpus-name` | Book | Approx. words |
| --------------- | ---- | ------------- |
| `alice`         | Alice's Adventures in Wonderland — Lewis Carroll       | ~28k |
| `frankenstein`  | Frankenstein — Mary Shelley                            | ~75k |
| `pride`         | Pride and Prejudice — Jane Austen                     | ~120k |
| `sherlock`      | The Adventures of Sherlock Holmes — A. Conan Doyle     | ~105k |
| `dorian`        | The Picture of Dorian Gray — Oscar Wilde               | ~78k |
| `moby`          | Moby-Dick — Herman Melville                           | ~215k |
| `warpeace`      | War and Peace — Leo Tolstoy                           | ~560k |
| `ulysses`       | Ulysses — James Joyce                                  | ~265k |

Gutenberg books are downloaded once and cached under
`backend/datasets/`. Use `--redownload` to force a refresh.

### Quick runs

```bash
# Default -- the bundled sample (always works, ~30s on a 3050)
python -m training.train --epochs 20

# Alice in Wonderland (good balance of size + variety)
python -m training.train --corpus gutenberg --corpus-name alice --epochs 25

# Pride and Prejudice (a bigger corpus, more variety)
python -m training.train --corpus gutenberg --corpus-name pride --epochs 30 \
  --max-vocab 8000 --batch-size 128

# Your own text file
python -m training.train --corpus file --corpus-path C:\path\to\my\book.txt \
  --epochs 30 --max-vocab 10000
```

### Tuning hyperparameters

```bash
python -m training.train \
  --corpus gutenberg --corpus-name alice \
  --epochs 30 \
  --batch-size 128 \
  --sequence-length 6 \
  --embedding-dim 128 \
  --hidden-dim 256 \
  --num-layers 2 \
  --dropout 0.3 \
  --learning-rate 5e-4
```

Typical training times on an RTX 3050 (4 GB):

| Corpus | Words | 20 epochs (3050) | 20 epochs (CPU) |
| ------ | ----- | ---------------- | --------------- |
| Bundled sample | 450 | ~30 s | ~2 min |
| Alice | 28k | ~1–2 min | ~10–15 min |
| Pride & Prejudice | 120k | ~4–6 min | ~30–45 min |
| War and Peace | 560k | ~20–30 min | ~2.5–3 h |

## Run the API

```powershell
cd C:\Users\soham\Desktop\Soham\GenAI\backend
.\.venv\Scripts\activate           # if not already active
uvicorn app.main:app --reload
```

- API root: <http://localhost:8000/>
- Swagger docs: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

## Endpoints

### `GET /`

Service banner.

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "device": "cpu"
}
```

### `GET /model-info`

```json
{
  "architecture": "LSTM",
  "framework": "PyTorch",
  "vocab_size": 12000,
  "embedding_dim": 128,
  "hidden_dim": 256,
  "num_layers": 2,
  "dropout": 0.2,
  "sequence_length": 5,
  "device": "cpu",
  "parameter_count": 2431784
}
```

### `POST /predict`

Request:

```json
{ "text": "Artificial intelligence is", "top_k": 5 }
```

Response:

```json
{
  "input_text": "Artificial intelligence is",
  "predictions": [
    { "word": "powerful", "probability": 0.724 },
    { "word": "changing", "probability": 0.118 }
  ]
}
```

### `POST /generate`

Request:

```json
{ "text": "Artificial intelligence is", "num_words": 5, "temperature": 1.0, "strategy": "greedy" }
```

Response:

```json
{
  "input_text": "Artificial intelligence is",
  "generated_text": "Artificial intelligence is changing the way we live",
  "generated_words": ["changing", "the", "way", "we", "live"]
}
```

## Tests

```powershell
cd C:\Users\soham\Desktop\Soham\GenAI\backend
.\.venv\Scripts\activate           # if not already active
pytest -q
```

The test suite creates a tiny randomly-initialized model and tokenizer
in a temp directory, patches the config paths, and exercises every
endpoint — including the failure paths (empty input, invalid `top_k`).
