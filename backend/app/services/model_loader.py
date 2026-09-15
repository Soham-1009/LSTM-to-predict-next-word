"""Load the trained LSTM model, tokenizer, and configuration exactly once.

`LifespanManager` is the single source of truth for whether the model is
available. API routes ask the manager for the current state and surface
`503` style errors when artifacts are missing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from ..config import CONFIG_PATH, DEVICE_ENV, MODEL_PATH, TOKENIZER_PATH
from ..models.lstm_model import LSTMConfig, LSTMPredictor
from ..utils.text_processing import Tokenizer

logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    if DEVICE_ENV == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE_ENV == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LoadedModel:
    model: LSTMPredictor
    tokenizer: Tokenizer
    config: LSTMConfig
    device: torch.device
    weights_path: str


class ModelManager:
    """Singleton-style holder for the loaded model and tokenizer."""

    def __init__(self) -> None:
        self._loaded: Optional[LoadedModel] = None

    # ------------------------------------------------------------------ props
    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def device_name(self) -> str:
        if self._loaded is None:
            return _resolve_device().type
        return self._loaded.device.type

    def config(self) -> Optional[LSTMConfig]:
        return None if self._loaded is None else self._loaded.config

    def tokenizer(self) -> Optional[Tokenizer]:
        return None if self._loaded is None else self._loaded.tokenizer

    def model(self) -> Optional[LSTMPredictor]:
        return None if self._loaded is None else self._loaded.model

    def parameter_count(self) -> int:
        if self._loaded is None:
            return 0
        return self._loaded.model.num_parameters()

    # ------------------------------------------------------------------ load
    def load(self) -> LoadedModel:
        """Load model, tokenizer, and config. Idempotent.

        Raises FileNotFoundError if a required artifact is missing.
        """
        if self._loaded is not None:
            return self._loaded

        # --- Validate files exist (don't fall back to a random stub) ------
        missing = [
            p
            for p, label in (
                (MODEL_PATH, "model weights"),
                (TOKENIZER_PATH, "tokenizer"),
                (CONFIG_PATH, "model config"),
            )
            if not os.path.exists(p)
        ]
        if missing:
            raise FileNotFoundError(
                "Required model artifacts are missing: "
                + ", ".join(f"{label} ({p})" for p, label in zip(missing, ("weights", "tokenizer", "config")))
            )

        # --- Config -------------------------------------------------------
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            raw_config = json.load(fh)
        config = LSTMConfig.from_dict(raw_config)

        # --- Tokenizer ---------------------------------------------------
        tokenizer = Tokenizer.load(str(TOKENIZER_PATH))

        # Sanity check: tokenizer vocabulary must match the config that was
        # saved alongside the model. If the user replaced the tokenizer file
        # we want to fail loudly rather than silently produce garbage.
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(
                f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match "
                f"config vocab size ({config.vocab_size})."
            )

        # --- Model --------------------------------------------------------
        device = _resolve_device()
        model = LSTMPredictor(config).to(device)

        try:
            state_dict = torch.load(str(MODEL_PATH), map_location=device)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load model weights: {exc}") from exc

        model.load_state_dict(state_dict)
        model.eval()

        self._loaded = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            weights_path=str(MODEL_PATH),
        )
        logger.info(
            "LSTM model loaded: vocab=%d embed=%d hidden=%d layers=%d seq=%d -> %s",
            config.vocab_size,
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            config.sequence_length,
            device,
        )
        return self._loaded

    def get(self) -> LoadedModel:
        """Return the loaded model, raising a clear error if not ready."""
        if self._loaded is None:
            raise RuntimeError(
                "LSTM model is not loaded. Run training first to produce "
                "model artifacts under backend/model/."
            )
        return self._loaded


# Process-wide singleton.
model_manager = ModelManager()
