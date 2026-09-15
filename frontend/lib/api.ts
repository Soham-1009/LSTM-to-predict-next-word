/**
 * Typed client for the FastAPI backend. Components import these
 * functions instead of calling `fetch` directly.
 *
 * Errors are normalized to `ApiError` so UI code can show a clean
 * message without having to parse different error shapes from each
 * endpoint.
 */

import { API_BASE_URL, REQUEST_TIMEOUT_MS } from "./config";
import type {
  GenerateRequest,
  GenerateResponse,
  GenerationStrategy,
  HealthResponse,
  ModelInfo,
  PredictRequest,
  PredictResponse,
} from "@/types/api";

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------
export class ApiError extends Error {
  public readonly status: number;
  public readonly detail: unknown;

  constructor(message: string, opts: { status: number; detail?: unknown } = { status: 0 }) {
    super(message);
    this.name = "ApiError";
    this.status = opts.status;
    this.detail = opts.detail;
  }
}

// ---------------------------------------------------------------------------
// Internal request helper
// ---------------------------------------------------------------------------
interface RequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, signal, timeoutMs = REQUEST_TIMEOUT_MS } = opts;
  const url = `${API_BASE_URL}${path}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // If the caller supplied their own signal, forward its abort.
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", () => controller.abort(), { once: true });
    }
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (networkError) {
    clearTimeout(timeoutId);
    if (controller.signal.aborted) {
      throw new ApiError("Request was cancelled or timed out.", { status: 0 });
    }
    throw new ApiError(
      `Could not reach the backend at ${API_BASE_URL}. Make sure FastAPI is running.`,
      { status: 0, detail: String(networkError) },
    );
  }
  clearTimeout(timeoutId);

  const text = await response.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }

  if (!response.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : `Request failed with status ${response.status}.`;
    throw new ApiError(message, { status: response.status, detail });
  }

  return data as T;
}

// ---------------------------------------------------------------------------
// Endpoint wrappers
// ---------------------------------------------------------------------------
function healthCheck(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { signal });
}

function modelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  return request<ModelInfo>("/model-info", { signal });
}

function predictNextWord(
  payload: { text: string; topK: number },
  signal?: AbortSignal,
): Promise<PredictResponse> {
  const body: PredictRequest = { text: payload.text, top_k: payload.topK };
  return request<PredictResponse>("/predict", { method: "POST", body, signal });
}

function generateText(
  payload: {
    text: string;
    numWords: number;
    temperature: number;
    strategy: GenerationStrategy;
  },
  signal?: AbortSignal,
): Promise<GenerateResponse> {
  const body: GenerateRequest = {
    text: payload.text,
    num_words: payload.numWords,
    temperature: payload.temperature,
    strategy: payload.strategy,
  };
  return request<GenerateResponse>("/generate", { method: "POST", body, signal });
}

export const api = {
  baseUrl: API_BASE_URL,
  healthCheck,
  modelInfo,
  predictNextWord,
  generateText,
};
