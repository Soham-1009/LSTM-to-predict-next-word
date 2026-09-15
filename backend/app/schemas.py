"""Pydantic request and response models used by the FastAPI layer.

Schemas deliberately mirror what the React frontend expects so the two ends
of the wire stay in lockstep.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Request body for next-word prediction."""

    text: str = Field(
        ...,
        description="Input sentence or text fragment. Must contain at least one word.",
        examples=["Artificial intelligence is"],
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="How many top predictions to return (1-20).",
    )

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        if value is None:
            raise ValueError("text is required")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text cannot be empty or whitespace only")
        return cleaned


class WordPrediction(BaseModel):
    """A single (word, probability) prediction."""

    word: str = Field(..., description="Predicted next word.")
    probability: float = Field(..., description="Softmax probability for this word.")


class PredictResponse(BaseModel):
    """Response body for next-word prediction."""

    input_text: str = Field(..., description="The original input text.")
    predictions: List[WordPrediction] = Field(
        ...,
        description="Top-k predictions sorted by probability in descending order.",
    )


# ---------------------------------------------------------------------------
# /generate
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """Request body for autoregressive text generation."""

    text: str = Field(
        ...,
        description="Seed text used to start generation.",
        examples=["Artificial intelligence is"],
    )
    num_words: int = Field(
        5,
        ge=1,
        le=50,
        description="How many words to generate (1-50).",
    )
    temperature: float = Field(
        1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature. Lower values are more deterministic.",
    )
    strategy: str = Field(
        "greedy",
        description="Sampling strategy: 'greedy' or 'top_k'.",
    )

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        if value is None:
            raise ValueError("text is required")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text cannot be empty or whitespace only")
        return cleaned

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"greedy", "top_k"}:
            raise ValueError("strategy must be either 'greedy' or 'top_k'")
        return normalized


class GenerateResponse(BaseModel):
    """Response body for autoregressive text generation."""

    input_text: str = Field(..., description="Original seed text.")
    generated_text: str = Field(..., description="Seed text + generated continuation.")
    generated_words: List[str] = Field(
        ...,
        description="List of words produced by the model in order.",
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' or 'unhealthy'.")
    model_loaded: bool = Field(..., description="Whether the LSTM weights are loaded.")
    tokenizer_loaded: bool = Field(..., description="Whether the tokenizer is loaded.")
    device: str = Field(..., description="Device used for inference (cpu or cuda).")


# ---------------------------------------------------------------------------
# /model-info
# ---------------------------------------------------------------------------
class ModelInfoResponse(BaseModel):
    architecture: str = Field(..., description="Model class name.")
    framework: str = Field(..., description="Deep learning framework used.")
    vocab_size: int = Field(..., description="Size of the vocabulary.")
    embedding_dim: int = Field(..., description="Embedding layer output dimension.")
    hidden_dim: int = Field(..., description="LSTM hidden state dimension.")
    num_layers: int = Field(..., description="Number of stacked LSTM layers.")
    dropout: float = Field(..., description="Dropout probability used during training.")
    sequence_length: int = Field(..., description="Input sequence length the model expects.")
    device: str = Field(..., description="Device used for inference.")
    parameter_count: Optional[int] = Field(
        None, description="Total number of trainable parameters in the model."
    )
