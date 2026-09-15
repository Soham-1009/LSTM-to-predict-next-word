# LSTM Next Word Predictor

A complete, college-project-ready demonstration of **next-word prediction
and text generation using a PyTorch LSTM**, served by a **FastAPI**
backend and a **Next.js 14 + TypeScript + Tailwind CSS** dashboard.

> The application does **not** call any external LLM. The predicted
> words come from a PyTorch LSTM that was trained locally on a public
> text corpus. Predictions are genuine softmax outputs of the trained
> model.

---

## 1. Project overview

| Layer    | Tech                              | Responsibility                                       |
| -------- | --------------------------------- | ---------------------------------------------------- |
| Frontend | Next.js 14 (App Router) + TS      | UI for entering text, showing predictions, history   |
| Styling  | Tailwind CSS 3                    | Modern academic AI dashboard look-and-feel           |
| Backend  | FastAPI (Python)                  | REST API: `/health`, `/predict`, `/generate`, `/model-info` |
| ML       | PyTorch 2.x                       | LSTM next-word model + tokenizer                     |
| Data     | Plain text files                  | Public-domain English corpus under `backend/datasets/` |

```
Next.js (App Router) + Tailwind CSS
        |
        | HTTP/REST JSON
        v
FastAPI Backend
        |
        +---- Prediction Service  (services/predictor.py)
        |        |
        |        v
        |     LSTM Model  (models/lstm_model.py)
        |
        +---- Tokenizer    (utils/text_processing.py)
        |
        +---- Model Configuration  (model/config.json)
```

---

## 2. Technologies

- **Python 3.10 or 3.11 (64-bit)** for the backend
- **FastAPI**, **Uvicorn**, **Pydantic v2**, **python-dotenv**
- **PyTorch 2.x**, **NumPy**
- **Node.js 18.17+ / 20+** for the frontend
- **Next.js 14** (App Router), **React 18**, **TypeScript**, **Tailwind CSS 3**
- **pytest**, **httpx** for backend tests
- **vitest**, **@testing-library/react** for frontend tests

---

## 3. Repository structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── api/routes.py
│   │   ├── services/
│   │   │   ├── model_loader.py
│   │   │   └── predictor.py
│   │   ├── models/lstm_model.py
│   │   └── utils/text_processing.py
│   ├── training/
│   │   ├── train.py
│   │   ├── preprocess.py
│   │   ├── dataset.py
│   │   └── evaluate.py
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_api.py
│   │   └── test_preprocessing.py
│   ├── datasets/sample_corpus.txt
│   ├── model/                 # produced by training
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── frontend/
│   ├── app/                       # Next.js App Router
│   │   ├── layout.tsx
│   │   ├── page.tsx               # / (dashboard)
│   │   ├── about/page.tsx         # /about
│   │   ├── loading.tsx
│   │   ├── error.tsx
│   │   ├── not-found.tsx
│   │   └── globals.css
│   ├── components/                # Feature-based organization
│   │   ├── dashboard/             # Header, TopNav, Dashboard
│   │   ├── prediction/            # PredictionPanel, PredictionResult
│   │   ├── generation/            # GenerationPanel, GenerationResult
│   │   ├── history/               # HistoryPanel
│   │   ├── model/                 # ModelInfoPanel
│   │   ├── education/             # HowItWorksPanel
│   │   └── ui/                    # Button, Card
│   ├── hooks/                     # useBackendStatus, usePrediction, useLocalHistory
│   ├── lib/                       # api.ts, config.ts, utils.ts
│   ├── types/                     # TypeScript mirrors of Pydantic schemas
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── postcss.config.mjs
│   ├── vitest.config.ts
│   ├── .eslintrc.json
│   ├── .env.example
│   └── README.md
├── .gitignore
└── README.md
```

---

## 4. LSTM theory (for the demonstration)

A standard LSTM cell is good at modeling sequences because it has:

1. **Cell state** — a memory conveyor that carries context across time
   steps.
2. **Three gates** — input, forget, output — that learn what to keep,
   what to discard, and what to expose.
3. **Hidden state** — the output summary at each time step, fed into
   the next time step.

For next-word prediction we feed the network the last *N* tokens, take
the hidden state at the final time step, and project it onto the
vocabulary through a linear layer. A softmax turns those logits into a
probability distribution; the highest-probability word is the model's
prediction.

Compared with vanilla RNNs, LSTMs mitigate the vanishing-gradient
problem and can learn longer-range dependencies, which is exactly the
property a next-word task benefits from.

---

## 5. How next-word prediction works here

1. The user types a sentence in the React UI.
2. The frontend POSTs the text to `/predict` (or `/generate`).
3. The backend normalizes the text, tokenizes it, and looks up word
   ids in the saved tokenizer.
4. The last `sequence_length` ids are wrapped in a `torch.Tensor` and
   passed through the trained LSTM under `torch.inference_mode()`.
5. The final hidden state is projected to vocabulary logits, softmax
   is applied, and the top-k words + probabilities are returned as
   JSON.
6. For multi-word generation, the predicted word is appended to the
   sequence and the model is run again — repeated `num_words` times.

---

## 6. Dataset

The training script loads `backend/datasets/sample_corpus.txt`. A
small public-domain sample is included so the project trains
immediately. To scale up, replace it with a larger Project Gutenberg
text or any other English corpus — the preprocessing pipeline in
`backend/training/preprocess.py` is corpus-agnostic.

| Property | Default value |
| -------- | ------------- |
| Source   | Bundled `sample_corpus.txt` (public domain) |
| Preprocess | normalize, lowercase, expand contractions, isolate punctuation, tokenize on whitespace |
| Vocab cutoff | configurable via `--min-freq` and `--max-vocab` |
| Special tokens | `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` |
| Sliding windows | `tokens[i : i+N] -> tokens[i+N]` |

---

## 7. End-to-end data flow

```
[React textarea]
   |
   | POST /predict { text, top_k }
   v
[FastAPI route]  --> Pydantic validation
   |
   v
[predict_next_word()]  --> normalize + tokenize + encode
   |
   v
[torch.Tensor (1, sequence_length)] --> LSTM
   |
   v
[logits (1, vocab_size)] --> softmax
   |
   v
[top-k (word, probability) pairs]
   |
   v
[JSON response] --> React renders cards + probability bars
```

For `/generate` the same loop runs `num_words` times, feeding each
predicted word back in.

---

## 8. Installation

### 8.0 Prerequisites

| Tool | Version | Notes |
| ---- | ------- | ----- |
| Python | **3.10 or 3.11 (64-bit)** | Do **not** use 3.12+ — the pinned `torch==2.3.0` and `numpy==1.26.4` wheels don't exist there. |
| Node.js | 18.17+ or 20+ | LTS recommended. |
| npm | 9+ (bundled with Node) | Or pnpm / yarn if you prefer. |
| (Optional) NVIDIA GPU + driver | CUDA 11.8 or 12.1+ | For training and inference on GPU. |

Install Python 3.11 from <https://www.python.org/downloads/windows/>.
On the installer's first screen, tick **both**:
- ☑ **Add python.exe to PATH**
- ☑ **Install Python 3.11 for all users** (this also installs the `py` launcher)

> The frontend (Node) and the backend (Python) are completely separate
> stacks. You only need **one** Python venv, and it lives inside
> `backend/`. The frontend's Node packages are managed by `npm` in their
> own `node_modules/` folder — no venv is needed there.

### 8.1 Backend (FastAPI + PyTorch)

```powershell
cd C:\Users\soham\Desktop\Soham\GenAI\backend

# Create the venv. If `python` is 3.11 already, this is enough.
# Otherwise use the launcher or a full path:
#   py -3.11 -m venv .venv
#   C:\Users\soham\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
python -m venv .venv
.\.venv\Scripts\activate
python --version          # MUST say 3.10.x or 3.11.x

# CPU-only stack
pip install -r requirements.txt

# OR GPU stack for an NVIDIA card 
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt

# Sanity check
python -c "import torch, fastapi, pydantic, numpy; print(torch.__version__, torch.cuda.is_available())"

# Optional: copy the env example
copy .env.example .env
```

### 8.2 Frontend (Next.js 14 + TypeScript)

```powershell
cd C:\Users\soham\Desktop\Soham\GenAI\frontend
npm install
copy .env.example .env.local       # optional; defaults to http://localhost:8000
```

> `NEXT_PUBLIC_API_BASE_URL` is the only frontend env var. The value is
> baked into the browser bundle at build time, so change it **before**
> `npm run build` if you need a different backend URL.

### 8.2 Frontend

```bash
cd frontend
npm install
cp .env.example .env.local       # optional; defaults to http://localhost:8000
```

---

## 9. Train the model

```powershell
cd C:\Users\soham\Desktop\Soham\GenAI\backend
.\.venv\Scripts\activate          # if not already active

# Default -- trains on the small bundled sample (always works, ~30s on a 3050)
python -m training.train --epochs 20

# Train on a real Project Gutenberg book (downloaded once, then cached)
python -m training.train --corpus gutenberg --corpus-name alice --epochs 25
python -m training.train --corpus gutenberg --corpus-name pride --epochs 30
python -m training.train --corpus gutenberg --corpus-name frankenstein --epochs 30

# Or supply your own text file
python -m training.train --corpus file --corpus-path C:\path\to\text.txt --epochs 30
```

Available Gutenberg books: `alice`, `frankenstein`, `pride`, `sherlock`, `dorian`, `moby`, `warpeace`, `ulysses`. Books are downloaded once and cached under `backend/datasets/`.

This produces four artifacts under `backend/model/`:

| File | Contents |
| ---- | -------- |
| `lstm_model.pth` | Trained LSTM weights (`state_dict`) |
| `tokenizer.pkl` | Fitted `Tokenizer` (word ↔ id) |
| `config.json` | Architecture & vocabulary config used to reconstruct the model |
| `training_history.json` | Per-epoch loss / accuracy / perplexity + corpus metadata |

> The backend refuses to start cleanly without these files. They are
> intentionally **not** committed to git because they are produced by
> training and can be large.

Useful flags:

```bash
python -m training.train \
  --corpus gutenberg --corpus-name alice \
  --epochs 30 --batch-size 128 \
  --sequence-length 6 \
  --embedding-dim 128 --hidden-dim 256 \
  --num-layers 2 --dropout 0.3 \
  --learning-rate 5e-4
```

Typical training times on an RTX 3050 (4 GB):

| Corpus | Words | 20 epochs |
| ------ | ----- | --------- |
| Bundled sample | 450 | ~30 s |
| Alice in Wonderland | 28k | ~1–2 min |
| Pride and Prejudice | 120k | ~4–6 min |
| War and Peace | 560k | ~20–30 min |

---

## 10. Start the API

```powershell
cd C:\Users\soham\Desktop\Soham\GenAI\backend
.\.venv\Scripts\activate           # if not already active
uvicorn app.main:app --reload
```

- API: <http://localhost:8000/>
- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

The `--reload` flag makes uvicorn restart on file changes — handy while
you tweak the API. If you want a production-style run, drop `--reload`.

---

## 11. Start the Next.js frontend

```powershell
cd C:\Users\soham\Desktop\Soham\GenAI\frontend
npm run dev
```

- Dashboard: <http://localhost:3000>
- About: <http://localhost:3000/about>
- Swagger (separate): <http://localhost:8000/docs>

The header shows a live backend status indicator that polls `/health`
every 30 seconds. The about page documents the stack and lists the API
endpoints with example responses.

---

## 12. API reference

### `GET /`

```json
{
  "name": "LSTM Next Word Predictor",
  "version": "1.0.0",
  "description": "...",
  "docs": "/docs"
}
```

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
    { "word": "changing", "probability": 0.118 },
    { "word": "transforming", "probability": 0.072 },
    { "word": "used", "probability": 0.049 },
    { "word": "becoming", "probability": 0.037 }
  ]
}
```

### `POST /generate`

Request:

```json
{
  "text": "Artificial intelligence is",
  "num_words": 5,
  "temperature": 1.0,
  "strategy": "greedy"
}
```

Response:

```json
{
  "input_text": "Artificial intelligence is",
  "generated_text": "Artificial intelligence is changing the way we live",
  "generated_words": ["changing", "the", "way", "we", "live"]
}
```

### Error responses

| Status | When |
| ------ | ---- |
| 400 | Empty input, invalid `top_k` / `num_words` / `strategy` |
| 503 | Model or tokenizer artifacts missing on disk |
| 500 | Catch-all for unexpected failures (no stack trace exposed) |

---

## 13. Frontend features

- Live backend status indicator in the header
- Text input with character counter, example prompts, clear & copy
- Single-word mode (top-k predictions) and multi-word generation
- Probability bars, top-1 word highlight, generated-continuation copy
- Expandable model information panel (architecture, vocab size, etc.)
- Expandable "How the model works" panel for academic walkthrough
- Recent-predictions history in `localStorage` with one-click re-use
- Loading / error / offline UI states everywhere

---

## 14. Running the tests

```powershell
# Backend
cd C:\Users\soham\Desktop\Soham\GenAI\backend
.\.venv\Scripts\activate
pytest -q
```

The test suite creates a small randomly-initialized LSTM + tokenizer
in a temp directory, patches the config paths, and exercises every
endpoint (including failure paths).

```powershell
# Frontend
cd C:\Users\soham\Desktop\Soham\GenAI\frontend
npm run test
```

The frontend suite uses **vitest** + **@testing-library/react** with
`jsdom`. Network calls are mocked, so the suite runs in seconds
without a live backend.

```powershell
# Full frontend check
npm run lint
npm run build
```

---

## 15. Troubleshooting

### Python / backend

| Problem | Fix |
| ------- | --- |
| `pip install -r requirements.txt` errors with "Could not find a version that satisfies the requirement torch" | Your venv is on Python 3.12+ where the pinned `torch==2.3.0` / `numpy==1.26.4` have no wheels. Recreate the venv with **Python 3.10 or 3.11** (download from <https://www.python.org/downloads/windows/>). Then `python --version` must show `3.10.x` or `3.11.x` **before** you `pip install`. |
| `py -3.11` says "is not recognized" | The `py` launcher wasn't installed. Re-run the Python 3.11 installer and tick **"Install Python 3.11 for all users"** (which also installs `py`). Or use the full path: `C:\Users\soham\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv`. |
| `python` resolves to a 3.14 install at the project root | Your system has a new Python but no 3.11. Install 3.11 alongside it, then either use the launcher (`py -3.11 -m venv .venv`) or the full path. |
| `/health` reports `model_loaded: false` | Run `python -m training.train` in `backend/` to produce the model artifacts. |
| "Tokenizer vocab size does not match config" | The model and tokenizer were trained separately. Re-run training. |
| Predictions look random | Either train for more epochs or supply a larger corpus (`--corpus gutenberg --corpus-name alice`). |

### Frontend (Next.js)

| Problem | Fix |
| ------- | --- |
| Dashboard shows "Backend offline" | Make sure FastAPI is running on the URL in `NEXT_PUBLIC_API_BASE_URL`. |
| CORS error in the browser console | Add your frontend origin to `ALLOWED_ORIGINS` in `backend/.env` and restart FastAPI. |
| `npm install` warns about peer deps | Safe to ignore. The pinned versions in `package.json` are compatible. |
| `npm run build` fails on TypeScript errors | Strict mode is on. `npx tsc --noEmit` to inspect the precise error. |
| Dashboard says "Backend connected" but Predict always fails | Open the browser dev tools → Network → click Predict. The response body's `detail` field tells you what's wrong (usually empty input or model not loaded). |

### venv reminders

- The venv lives at `backend/.venv/`. Do not create a second one in
  the project root.
- The frontend has **no** Python venv — it's pure Node.js. Run
  `npm install` once and you're done.

---

## 16. Future improvements

- Swap greedy/top-k sampling for nucleus (top-p) sampling
- Train on a larger Project Gutenberg corpus
- Add attention visualization for the LSTM hidden states
- Add unit tests for the React components (Vitest + Testing Library)
- Containerize the stack with Docker / docker-compose
