"""Pydantic schemas for FastAPI endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    device: str = "cpu"


class ModelInfoResponse(BaseModel):
    model_name: str = "LSTM Causal Language Model"
    architecture: str = "2-Layer LSTM"
    vocab_size: int = 30000
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    context_length: int = 35
    device: str = "cpu"
    checkpoint_loaded: bool = False
    active_checkpoint: str = "baseline"
    available_checkpoints: List[str] = ["baseline", "adapted"]
    status: str = "ready"


class SelectCheckpointRequest(BaseModel):
    checkpoint_name: str = Field(..., description="Name of checkpoint to switch to: 'baseline' or 'adapted'")


class SelectCheckpointResponse(BaseModel):
    active_checkpoint: str
    checkpoint_path: str
    status: str


class CandidateToken(BaseModel):
    token: str
    probability: float = Field(..., ge=0.0, le=1.0)
    token_id: int


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Context sequence for next-word prediction")
    top_k: int = Field(5, ge=1, le=50, description="Number of candidate suggestions to return")


class PredictResponse(BaseModel):
    input_text: str
    predictions: List[CandidateToken]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt to begin autoregressive text generation")
    max_new_tokens: int = Field(30, ge=1, le=250, description="Maximum number of tokens to generate")
    temperature: float = Field(1.0, ge=0.0, le=5.0, description="Sampling temperature (lower = more deterministic)")
    top_k: Optional[int] = Field(40, ge=1, description="Top-k filtering limit")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling threshold")
    greedy: bool = Field(False, description="Whether to use greedy decoding")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int
    finish_reason: str = "length"
