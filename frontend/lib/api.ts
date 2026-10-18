/**
 * API client for LSTM Language Model backend service.
 */

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const saved = localStorage.getItem("lstm_api_base_url");
    if (saved) return saved;
  }
  return process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
}

export function setApiBaseUrl(url: string): void {
  if (typeof window !== "undefined") {
    const trimmed = url.trim().replace(/\/+$/, "");
    if (trimmed) {
      localStorage.setItem("lstm_api_base_url", trimmed);
    } else {
      localStorage.removeItem("lstm_api_base_url");
    }
  }
}

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
}

export interface ModelInfoResponse {
  model_name: string;
  architecture: string;
  vocab_size: number;
  embed_dim: number;
  hidden_dim: number;
  num_layers: number;
  dropout: number;
  context_length: number;
  device: string;
  checkpoint_loaded: boolean;
  active_checkpoint?: string;
  available_checkpoints?: string[];
  status: string;
}

export interface DomainMetrics {
  loss: number;
  perplexity: number;
  top1_acc: number;
  top5_acc: number;
}

export interface DomainComparisonItem {
  domain: string;
  domain_type: string;
  document_count: number;
  tokens: number;
  evaluated_tokens?: number;
  oov_rate_pct: number;
  vocab_coverage_pct: number;
  baseline: DomainMetrics;
  adapted: DomainMetrics;
  deltas: {
    loss_delta: number;
    ppl_delta: number;
    top1_delta: number;
    top3_delta?: number;
    top5_delta: number;
    improved: boolean;
    regressed?: boolean;
    notes?: string | null;
  };
}

export interface QualitativeComparisonItem {
  domain: string;
  prompt: string;
  baseline_greedy: string;
  adapted_greedy: string;
  baseline_sampling: string;
  adapted_sampling: string;
}

export interface GeneralizationReport {
  title: string;
  description: string;
  benchmark_scope?: {
    type: string;
    document_count: number;
    domains_evaluated: number;
    total_tokens: number;
    evaluated_target_tokens: number;
    causal_window_size: number;
    document_boundary_safe: boolean;
  };
  timestamp_utc: string;
  baseline_metadata: {
    checkpoint: string;
    architecture: string;
    primary_training: string;
  };
  adapted_metadata: {
    checkpoint: string;
    fine_tuning: string;
    vocabulary: string;
  };
  leakage_audit_summary: {
    status: string;
    invariants_passed: string;
    parent_source_overlap_pct: number;
    clean_hash_overlap_pct: number;
    document_boundary_safe: boolean;
    audit_script: string;
  };
  full_test_baseline_reference?: any;
  overall_summary: {
    weighting_type?: string;
    baseline: any;
    adapted: any;
    deltas: {
      loss_delta: number;
      ppl_delta: number;
      ppl_pct_change: number;
      top1_delta: number;
      top3_delta?: number;
      top5_delta: number;
    };
  };
  token_weighted_summary?: {
    weighting_type: string;
    description: string;
    baseline: any;
    adapted: any;
    deltas: any;
  };
  macro_average_summary?: {
    weighting_type: string;
    description: string;
    baseline: {
      domain_count: number;
      loss: number;
      perplexity: number;
      geometric_perplexity?: number;
      top1_acc: number;
      top3_acc: number;
      top5_acc: number;
    };
    adapted: {
      domain_count: number;
      loss: number;
      perplexity: number;
      geometric_perplexity?: number;
      top1_acc: number;
      top3_acc: number;
      top5_acc: number;
    };
    deltas: {
      loss_delta: number;
      ppl_delta: number;
      ppl_pct_change: number;
      geometric_ppl_delta?: number;
      geometric_ppl_pct_change?: number;
      top1_delta: number;
      top3_delta?: number;
      top5_delta: number;
    };
  };
  in_domain_retention: {
    baseline: any;
    adapted: any;
    deltas: {
      loss_delta: number;
      ppl_delta: number;
      ppl_pct_change?: number;
      top1_delta: number;
      top5_delta: number;
      retention_status?: string;
      description?: string;
      catastrophic_forgetting?: boolean;
    };
  };
  domain_breakdown: DomainComparisonItem[];
  qualitative_comparison: QualitativeComparisonItem[];
  metric_limitations: Record<string, string>;
}

export interface SelectCheckpointResponse {
  active_checkpoint: string;
  checkpoint_path: string;
  status: string;
}

export interface CandidateToken {
  token: string;
  probability: number;
  token_id: number;
}

export interface PredictRequest {
  text: string;
  top_k?: number;
}

export interface PredictResponse {
  input_text: string;
  predictions: CandidateToken[];
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  greedy?: boolean;
}

export interface GenerateResponse {
  prompt: string;
  generated_text: string;
  tokens_generated: number;
  finish_reason: string;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/health`, { cache: "no-store" });
    if (res.ok) return await res.json();
  } catch {
    // Offline
  }
  return {
    status: "offline",
    version: "0.1.0",
    device: "CPU",
  };
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/model-info`, { cache: "no-store" });
    if (res.ok) return await res.json();
  } catch {
    // Offline metadata fallback
  }
  return {
    model_name: "LSTM Causal Language Model",
    architecture: "2-Layer LSTM (embedding 256, hidden 512, vocab 30000, context 35)",
    vocab_size: 30000,
    embed_dim: 256,
    hidden_dim: 512,
    num_layers: 2,
    dropout: 0.2,
    context_length: 35,
    device: "CPU",
    checkpoint_loaded: false,
    active_checkpoint: "best_val.pt",
    available_checkpoints: ["best_val.pt", "multidomain_adapted.pt"],
    status: "degraded",
  };
}

function getFallbackPredictions(text: string, topK: number): CandidateToken[] {
  const normalized = text.trim().toLowerCase().replace(/[^a-z0-9\s]/g, "");

  if (normalized.endsWith("once upon a") || normalized === "once upon a") {
    return [
      { token: "time", probability: 0.684, token_id: 61 },
      { token: "midnight", probability: 0.082, token_id: 1420 },
      { token: "day", probability: 0.061, token_id: 112 },
      { token: "hill", probability: 0.045, token_id: 840 },
      { token: "summer", probability: 0.032, token_id: 421 },
    ].slice(0, topK);
  }

  if (normalized.endsWith("history of the") || normalized.endsWith("history of the modern")) {
    return [
      { token: "world", probability: 0.412, token_id: 42 },
      { token: "church", probability: 0.184, token_id: 195 },
      { token: "state", probability: 0.142, token_id: 88 },
      { token: "first", probability: 0.098, token_id: 35 },
      { token: "city", probability: 0.054, token_id: 120 },
    ].slice(0, topK);
  }

  if (normalized.endsWith("in computer science a") || normalized.endsWith("computer science a")) {
    return [
      { token: "function", probability: 0.385, token_id: 280 },
      { token: "program", probability: 0.241, token_id: 310 },
      { token: "system", probability: 0.154, token_id: 75 },
      { token: "set", probability: 0.089, token_id: 140 },
      { token: "process", probability: 0.062, token_id: 220 },
    ].slice(0, topK);
  }

  if (normalized.endsWith("demonstrate that") || normalized.endsWith("results demonstrate that")) {
    return [
      { token: "the", probability: 0.542, token_id: 1 },
      { token: "this", probability: 0.165, token_id: 21 },
      { token: "a", probability: 0.112, token_id: 6 },
      { token: "patients", probability: 0.064, token_id: 890 },
      { token: "models", probability: 0.048, token_id: 450 },
    ].slice(0, topK);
  }

  if (normalized.endsWith("into the quiet") || normalized.endsWith("the quiet")) {
    return [
      { token: "room", probability: 0.512, token_id: 185 },
      { token: "night", probability: 0.194, token_id: 240 },
      { token: "street", probability: 0.125, token_id: 310 },
      { token: "water", probability: 0.068, token_id: 190 },
      { token: "corner", probability: 0.042, token_id: 420 },
    ].slice(0, topK);
  }

  if (normalized.endsWith("can you tell me") || normalized.endsWith("tell me")) {
    return [
      { token: "what", probability: 0.442, token_id: 39 },
      { token: "how", probability: 0.221, token_id: 52 },
      { token: "about", probability: 0.148, token_id: 44 },
      { token: "if", probability: 0.085, token_id: 43 },
      { token: "where", probability: 0.052, token_id: 70 },
    ].slice(0, topK);
  }

  const words = normalized.split(/\s+/).filter(Boolean);
  const last = words[words.length - 1] || "";

  const fallbacks: Record<string, string[]> = {
    the: ["first", "world", "same", "new", "time"],
    of: ["the", "a", "this", "his", "their"],
    to: ["the", "be", "a", "make", "have"],
    in: ["the", "a", "this", "which", "its"],
    is: ["a", "the", "not", "an", "used"],
    that: ["the", "he", "it", "they", "a"],
    a: ["new", "small", "large", "single", "good"],
    and: ["the", "he", "she", "a", "was"],
    for: ["the", "a", "example", "his", "this"],
    on: ["the", "a", "his", "this", "board"],
  };

  const list = fallbacks[last] || ["the", "and", "to", "of", "a"];
  const probs = [0.42, 0.24, 0.16, 0.10, 0.08];

  return list.slice(0, topK).map((token, idx) => ({
    token,
    probability: probs[idx] ?? 0.05,
    token_id: 100 + idx,
  }));
}

export async function predictNextWord(request: PredictRequest): Promise<PredictResponse> {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: request.text,
        top_k: request.top_k ?? 5,
      }),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // If backend is unreachable on GitHub Pages, return offline demo candidate predictions
    return {
      input_text: request.text,
      predictions: getFallbackPredictions(request.text, request.top_k ?? 5),
    };
  }

  return {
    input_text: request.text,
    predictions: getFallbackPredictions(request.text, request.top_k ?? 5),
  };
}

export async function generateText(request: GenerateRequest): Promise<GenerateResponse> {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: request.prompt,
        max_new_tokens: request.max_new_tokens ?? 30,
        temperature: request.temperature ?? 1.0,
        top_k: request.top_k ?? 40,
        top_p: request.top_p ?? 0.9,
        greedy: request.greedy ?? false,
      }),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Offline demo continuation fallback
  }

  const promptTrimmed = request.prompt.trim();
  let continuation = "";
  if (promptTrimmed.toLowerCase().includes("once upon a")) {
    continuation = "time, in a quiet village surrounded by ancient woods, there lived an old storyteller whose words could captivate any traveler.";
  } else if (promptTrimmed.toLowerCase().includes("history of the")) {
    continuation = "modern world was shaped by political revolutions, technological advances, and international trade networks expanding across continents.";
  } else if (promptTrimmed.toLowerCase().includes("computer science")) {
    continuation = "function is a self-contained block of code designed to perform a specific task, taking inputs and returning output values.";
  } else {
    continuation = "first step demonstrated significant stability across subsequent iterations, providing strong foundation for further experimental validation.";
  }

  const fullText = promptTrimmed ? `${promptTrimmed} ${continuation}` : continuation;
  const wordCount = continuation.split(/\s+/).filter(Boolean).length;

  return {
    prompt: request.prompt,
    generated_text: fullText,
    tokens_generated: wordCount,
    finish_reason: "length",
  };
}

export async function fetchGeneralizationReport(): Promise<GeneralizationReport> {
  const base = getApiBaseUrl();
  try {
    const res = await fetch(`${base}/evaluation/generalization`, { cache: "no-store" });
    if (res.ok) return await res.json();
  } catch {
    // API unreachable, fall back to bundled static artifact for static/GitHub Pages deployment
  }

  const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
  const staticUrl = `${basePath}/benchmark/generalization_comparison.json`;
  try {
    const staticRes = await fetch(staticUrl);
    if (staticRes.ok) return await staticRes.json();
  } catch {
    // Both failed
  }

  throw new Error("Could not load generalization benchmark data from API or bundled report.");
}

export async function fetchTestBaselineReport(): Promise<any> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/evaluation/test-baseline`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch baseline report: ${res.statusText}`);
  return res.json();
}

export async function selectCheckpoint(checkpointName: string): Promise<SelectCheckpointResponse> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/model/select-checkpoint`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint_name: checkpointName }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to switch checkpoint: ${res.statusText}`);
  }
  return res.json();
}
