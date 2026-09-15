/**
 * TypeScript mirrors of the FastAPI/Pydantic schemas in
 * `backend/app/schemas.py`. Keep these in lockstep with the backend
 * whenever endpoints change.
 */

// ---------------------------------------------------------------------------
// /health
// ---------------------------------------------------------------------------
export type HealthState = "online" | "degraded" | "offline" | "checking";

export interface HealthResponse {
  status: "healthy" | "degraded";
  model_loaded: boolean;
  tokenizer_loaded: boolean;
  device: string;
}

// ---------------------------------------------------------------------------
// /model-info
// ---------------------------------------------------------------------------
export interface ModelInfo {
  architecture: string;
  framework: string;
  vocab_size: number;
  embedding_dim: number;
  hidden_dim: number;
  num_layers: number;
  dropout: number;
  sequence_length: number;
  device: string;
  parameter_count: number | null;
}

// ---------------------------------------------------------------------------
// /predict
// ---------------------------------------------------------------------------
export interface WordPrediction {
  word: string;
  probability: number;
}

export interface PredictRequest {
  text: string;
  top_k: number;
}

export interface PredictResponse {
  input_text: string;
  predictions: WordPrediction[];
}

// ---------------------------------------------------------------------------
// /generate
// ---------------------------------------------------------------------------
export type GenerationStrategy = "greedy" | "top_k";

export interface GenerateRequest {
  text: string;
  num_words: number;
  temperature: number;
  strategy: GenerationStrategy;
}

export interface GenerateResponse {
  input_text: string;
  generated_text: string;
  generated_words: string[];
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------
export type HistoryKind = "predict" | "generate";

export interface HistoryEntry {
  id: string;
  ts: number;
  kind: HistoryKind;
  input: string;
  output: string;
  confidence?: number;
  generated_words?: string[];
}
