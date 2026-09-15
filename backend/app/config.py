"""Application configuration loaded from environment variables.

All knobs (model paths, device selection, CORS origins) are exposed via
environment variables. Defaults are tuned for local development.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env if present; safe to call multiple times.
load_dotenv()


# Resolve base directory so paths are stable regardless of cwd.
BASE_DIR: Path = Path(__file__).resolve().parent.parent


def _resolve_path(value: str, default_relative: str) -> Path:
    """Return absolute Path for a setting; supports absolute or relative values."""
    p = Path(value)
    if not p.is_absolute():
        p = BASE_DIR / default_relative
    return p


def _split_origins(value: str) -> List[str]:
    """Split a comma-separated origin string, stripping whitespace."""
    return [origin.strip() for origin in value.split(",") if origin.strip()]


# --- Model & tokenizer artifacts --------------------------------------------
MODEL_PATH: Path = _resolve_path(
    os.getenv("MODEL_PATH", "model/lstm_model.pth"),
    "model/lstm_model.pth",
)
TOKENIZER_PATH: Path = _resolve_path(
    os.getenv("TOKENIZER_PATH", "model/tokenizer.pkl"),
    "model/tokenizer.pkl",
)
CONFIG_PATH: Path = _resolve_path(
    os.getenv("CONFIG_PATH", "model/config.json"),
    "model/config.json",
)

# --- Device ------------------------------------------------------------------
DEVICE_ENV: str = os.getenv("DEVICE", "auto").lower()
# Resolution happens lazily inside services.model_loader to avoid importing torch
# at module import time when only config is needed (e.g., in tooling).

# --- CORS --------------------------------------------------------------------
# Allowed origins for the frontend.  Includes the Vite dev server (5173) for
# legacy compatibility and the Next.js dev server (3000) for the new frontend.
ALLOWED_ORIGINS: List[str] = _split_origins(
    os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000",
    )
)

# --- Limits ------------------------------------------------------------------
MAX_INPUT_CHARS: int = int(os.getenv("MAX_INPUT_CHARS", "500"))
MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "20"))
MAX_GENERATE_WORDS: int = int(os.getenv("MAX_GENERATE_WORDS", "50"))

# --- API metadata ------------------------------------------------------------
APP_TITLE: str = "LSTM Next Word Predictor"
APP_DESCRIPTION: str = (
    "Generative AI project that performs next-word prediction using a "
    "trained PyTorch LSTM neural network."
)
APP_VERSION: str = "1.0.0"
