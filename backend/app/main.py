"""FastAPI application for LSTM Language Model inference.

Exposes REST endpoints:
- GET /health
- GET /model-info
- POST /predict (Real tokenization -> LSTM forward -> Logits -> Softmax candidate ranking)
- POST /generate (Real tokenization -> Autoregressive LSTM generation loop -> Sampling -> Decoding)
"""

from contextlib import asynccontextmanager
from pathlib import Path
import json
import threading
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.app.schemas import (
        HealthResponse,
        ModelInfoResponse,
        PredictRequest,
        PredictResponse,
        GenerateRequest,
        GenerateResponse,
        CandidateToken,
        SelectCheckpointRequest,
        SelectCheckpointResponse,
    )
    from backend.model.config import ModelConfig
    from backend.model.lstm import LSTMLanguageModel, sample_next_token
    from backend.model.tokenizer import Tokenizer
except ImportError:
    from app.schemas import (
        HealthResponse,
        ModelInfoResponse,
        PredictRequest,
        PredictResponse,
        GenerateRequest,
        GenerateResponse,
        CandidateToken,
        SelectCheckpointRequest,
        SelectCheckpointResponse,
    )
    from model.config import ModelConfig
    from model.lstm import LSTMLanguageModel, sample_next_token
    from model.tokenizer import Tokenizer

# Hardware device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initial starter vocabulary so the tokenization and decoding pipeline functions immediately
STARTER_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
    "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new",
    "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was",
    "were", "has", "had", "been", "system", "model", "learning", "data", "deep", "language",
    "text", "neural", "network", "generation", "prediction", "corpus", "token", "sequence",
    "world", "machine", "algorithm", "intelligence", "artificial", "pipeline", "weights"
]


class InferenceService:
    """Manages model and tokenizer lifecycles for inference serving."""

    def __init__(self) -> None:
        self.config = ModelConfig()
        self.tokenizer = Tokenizer()
        # Seed vocabulary
        freqs = {word: 1000 - i for i, word in enumerate(STARTER_WORDS)}
        self.tokenizer.build_vocab(freqs, max_vocab_size=self.config.vocab_size)

        self.model: Optional[LSTMLanguageModel] = None
        self.checkpoint_loaded: bool = False
        self.active_checkpoint_name: str = "baseline"
        self.device = DEVICE
        self._init_lock = threading.Lock()
        self._model_lock = threading.Lock()

    def resolve_checkpoint_path(self, checkpoint_name: str) -> tuple[str, Path]:
        """Resolve path to checkpoint file, supporting both canonical names and filename aliases."""
        norm_name = checkpoint_name.lower().strip()
        if norm_name in ("baseline", "best_val.pt"):
            canonical_name = "baseline"
            ckpt_path = Path("backend/model_artifacts/checkpoints/best_val.pt")
            if not ckpt_path.exists():
                candidate = Path(__file__).resolve().parent.parent / "model_artifacts" / "checkpoints" / "best_val.pt"
                if candidate.exists():
                    ckpt_path = candidate
        elif norm_name in ("adapted", "multidomain_adapted.pt"):
            canonical_name = "adapted"
            ckpt_path = Path("experiments/multidomain_v1/checkpoints/multidomain_adapted.pt")
            if not ckpt_path.exists():
                candidate = Path(__file__).resolve().parent.parent.parent / "experiments" / "multidomain_v1" / "checkpoints" / "multidomain_adapted.pt"
                if candidate.exists():
                    ckpt_path = candidate
        else:
            raise ValueError(f"Unknown checkpoint '{checkpoint_name}'. Choose 'baseline' or 'adapted'.")
        return canonical_name, ckpt_path

    def load_checkpoint(self, checkpoint_name: str) -> str:
        """Load specified checkpoint weights into current model architecture.

        Thread-safe: constructs new model instance and atomically swaps self.model,
        preventing in-place parameter mutation during concurrent generation runs.
        """
        canonical_name, ckpt_path = self.resolve_checkpoint_path(checkpoint_name)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        vocab_size = max(len(self.tokenizer), self.config.vocab_size)
        new_model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            pad_idx=self.tokenizer.pad_id,
        ).to(self.device)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        new_model.eval()

        with self._model_lock:
            self.model = new_model
            self.active_checkpoint_name = canonical_name
            self.checkpoint_loaded = True
        return str(ckpt_path)

    def initialize(self) -> None:
        """Initialize the LSTM model architecture."""
        # Attempt to load saved tokenizer artifact if present
        vocab_path = Path("backend/model_artifacts/tokenizer/vocab.json")
        if not vocab_path.exists():
            backend_dir = Path(__file__).resolve().parent.parent
            candidate = backend_dir / "model_artifacts" / "tokenizer" / "vocab.json"
            if candidate.exists():
                vocab_path = candidate

        if vocab_path.exists():
            try:
                self.tokenizer = Tokenizer.load(vocab_path)
            except Exception as e:
                print(f"Warning: could not load {vocab_path}: {e}")

        # Update vocab size from tokenizer
        vocab_size = max(len(self.tokenizer), self.config.vocab_size)
        self.model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            pad_idx=self.tokenizer.pad_id,
        ).to(self.device)
        self.model.eval()

        # Attempt to load baseline checkpoint weights
        try:
            loaded_path = self.load_checkpoint("baseline")
            print(f"Loaded baseline model weights from {loaded_path}")
        except Exception as e:
            print(f"Warning: could not load baseline checkpoint: {e}")

    def ensure_initialized(self) -> None:
        """Ensure model and tokenizer are initialized even if lifespan was bypassed."""
        if self.model is None:
            with self._init_lock:
                if self.model is None:
                    self.initialize()


service = InferenceService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context."""
    service.initialize()
    yield


app = FastAPI(
    title="LSTM Next-Word Prediction & Text Generation API",
    description="FastAPI layer for tokenization, causal next-word prediction, and autoregressive text generation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return health status and active compute device."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        device=str(service.device),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Return current model configuration and deployment state."""
    service.ensure_initialized()
    if service.model is None:
        status_str = "initializing"
    elif service.checkpoint_loaded:
        status_str = "ready"
    else:
        status_str = "degraded"

    return ModelInfoResponse(
        model_name="LSTM Causal Language Model",
        architecture="2-Layer LSTM",
        vocab_size=len(service.tokenizer),
        embed_dim=service.config.embed_dim,
        hidden_dim=service.config.hidden_dim,
        num_layers=service.config.num_layers,
        dropout=service.config.dropout,
        context_length=service.config.context_length,
        device=str(service.device),
        checkpoint_loaded=service.checkpoint_loaded,
        active_checkpoint=service.active_checkpoint_name,
        available_checkpoints=["baseline", "adapted"],
        status=status_str,
    )


@app.post("/model/select-checkpoint", response_model=SelectCheckpointResponse)
def select_checkpoint(request: SelectCheckpointRequest) -> SelectCheckpointResponse:
    """Switch active checkpoint between baseline (best_val.pt) and adapted (multidomain_adapted.pt)."""
    service.ensure_initialized()
    try:
        path = service.load_checkpoint(request.checkpoint_name)
        return SelectCheckpointResponse(
            active_checkpoint=service.active_checkpoint_name,
            checkpoint_path=path,
            status="success",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {e}")


@app.get("/evaluation/generalization")
def get_generalization_report() -> Dict[str, Any]:
    """Return verified empirical multi-domain evaluation comparison report."""
    candidates = [
        Path("backend/model_artifacts/results/generalization_comparison.json"),
        Path(__file__).resolve().parent.parent / "model_artifacts" / "results" / "generalization_comparison.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail="Generalization comparison report not found.")


@app.get("/evaluation/test-baseline")
def get_test_baseline_report() -> Dict[str, Any]:
    """Return verified baseline evaluation report on full WikiText-103 test split."""
    candidates = [
        Path("backend/model_artifacts/results/test_evaluation_epoch46.json"),
        Path(__file__).resolve().parent.parent / "model_artifacts" / "results" / "test_evaluation_epoch46.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail="Baseline test evaluation report not found.")


@app.post("/predict", response_model=PredictResponse)
def predict_next_word(request: PredictRequest) -> PredictResponse:
    """Tokenize input context, execute LSTM forward pass, and return candidate predictions."""
    service.ensure_initialized()
    trimmed = request.text.strip()
    if not trimmed:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if service.model is None:
        raise HTTPException(status_code=503, detail="Model is initializing.")

    # 1. Tokenize input text to token IDs
    token_ids = service.tokenizer.encode(trimmed)
    if not token_ids:
        token_ids = [service.tokenizer.unk_id]

    # Truncate to context length
    context_tokens = token_ids[-service.config.context_length :]
    input_tensor = torch.tensor([context_tokens], dtype=torch.long, device=service.device)

    # 2. Forward pass through LSTM
    try:
        with service._model_lock:
            model_ref = service.model
        with torch.inference_mode():
            logits, _ = model_ref(input_tensor)
            # Extract logits for the final step: shape (vocab_size,)
            last_logits = logits[0, -1, :].clone()

            # Zero out control tokens so they are not offered as autocomplete suggestions (FR-12)
            control_ids = [
                service.tokenizer.pad_id,
                service.tokenizer.unk_id,
                service.tokenizer.bos_id,
                service.tokenizer.eos_id,
            ]
            last_logits = torch.nan_to_num(last_logits, nan=-float("inf"), posinf=1e9, neginf=-float("inf"))
            for cid in control_ids:
                if cid < last_logits.size(0):
                    last_logits[cid] = -float("Inf")

            probs = F.softmax(last_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            prob_sum = probs.sum().item()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                valid_mask = torch.ones_like(probs)
                for cid in control_ids:
                    if cid < valid_mask.size(0):
                        valid_mask[cid] = 0.0
                probs = valid_mask / max(valid_mask.sum().item(), 1.0)

            # 3. Top-K ranking
            top_k = min(request.top_k, probs.size(0))
            top_probs, top_indices = torch.topk(probs, top_k)

        candidates: List[CandidateToken] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token_str = service.tokenizer.id_to_token.get(idx, f"token_{idx}")
            candidates.append(
                CandidateToken(
                    token=token_str,
                    probability=round(float(prob), 4),
                    token_id=idx,
                )
            )

        return PredictResponse(
            input_text=request.text,
            predictions=candidates,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Inference processing failed.")


@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest) -> GenerateResponse:
    """Perform autoregressive token generation using sampling controls."""
    service.ensure_initialized()
    trimmed = request.prompt.strip()
    if not trimmed:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if service.model is None:
        raise HTTPException(status_code=503, detail="Model is initializing.")

    # 1. Tokenize seed prompt
    token_ids = service.tokenizer.encode(trimmed)
    if not token_ids:
        token_ids = [service.tokenizer.bos_id]

    try:
        generated_ids: List[int] = list(token_ids)
        finish_reason = "length"

        # 2. Autoregressive generation loop
        with service._model_lock:
            model_ref = service.model
        with torch.inference_mode():
            for _ in range(request.max_new_tokens):
                context = generated_ids[-service.config.context_length :]
                inp = torch.tensor([context], dtype=torch.long, device=service.device)
                logits, _ = model_ref(inp)
                step_logits = logits[0, -1, :].clone()
                step_logits = torch.nan_to_num(step_logits, nan=-float("inf"), posinf=1e9, neginf=-float("inf"))

                # Prevent predicting PAD, BOS, and UNK during generation to ensure natural text output
                for cid in [
                    service.tokenizer.pad_id,
                    service.tokenizer.bos_id,
                    service.tokenizer.unk_id,
                ]:
                    if cid < step_logits.size(0):
                        step_logits[cid] = -float("Inf")

                # Sample next token
                next_id = sample_next_token(
                    step_logits,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    greedy=request.greedy,
                )

                # Check EOS termination (FR-13)
                if next_id == service.tokenizer.eos_id:
                    finish_reason = "eos"
                    break

                generated_ids.append(next_id)

        # 3. Decode token sequence back to text
        full_text = service.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_produced = len(generated_ids) - len(token_ids)

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=full_text,
            tokens_generated=tokens_produced,
            finish_reason=finish_reason,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Generation processing failed.")
