"""FastAPI route definitions.

The router is intentionally thin: it validates input with Pydantic, delegates
the actual ML work to `services.predictor`, and translates exceptions into
appropriate HTTP responses.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from .. import config
from ..schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    WordPrediction,
)
from ..services.model_loader import model_manager
from ..services.predictor import (
    generate_text,
    is_input_too_short,
    model_metadata,
    predict_next_word,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /  --  root metadata
# ---------------------------------------------------------------------------
@router.get("/", summary="Service banner")
def root() -> dict:
    return {
        "name": config.APP_TITLE,
        "version": config.APP_VERSION,
        "description": config.APP_DESCRIPTION,
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, summary="Health probe")
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        tokenizer_loaded=model_manager.tokenizer() is not None,
        device=model_manager.device_name(),
    )


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------
@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Architecture & vocabulary metadata",
)
def model_info() -> ModelInfoResponse:
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LSTM model is not available. Train the model first.",
        )
    try:
        meta = model_metadata()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return ModelInfoResponse(**meta)


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict the top-k most likely next words",
)
def predict(payload: PredictRequest) -> PredictResponse:
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LSTM model is not available. Train the model first.",
        )
    if is_input_too_short(payload.text, sequence_length=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text must contain at least one word.",
        )

    try:
        results = predict_next_word(
            text=payload.text,
            top_k=payload.top_k,
        )
    except RuntimeError as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction.",
        ) from exc

    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model could not produce predictions for this input.",
        )

    return PredictResponse(
        input_text=payload.text,
        predictions=[
            WordPrediction(word=p.word, probability=round(p.probability, 6))
            for p in results
        ],
    )


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------
@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Autoregressively generate a continuation",
)
def generate(payload: GenerateRequest) -> GenerateResponse:
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LSTM model is not available. Train the model first.",
        )
    if is_input_too_short(payload.text, sequence_length=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text must contain at least one word.",
        )

    try:
        words = generate_text(
            text=payload.text,
            num_words=payload.num_words,
            temperature=payload.temperature,
            strategy=payload.strategy,
        )
    except RuntimeError as exc:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected generation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate text.",
        ) from exc

    generated_text = (payload.text.rstrip() + " " + " ".join(words)).strip()
    return GenerateResponse(
        input_text=payload.text,
        generated_text=generated_text,
        generated_words=words,
    )
