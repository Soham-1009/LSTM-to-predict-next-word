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

export async function predictNextWord(request: PredictRequest): Promise<PredictResponse> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: request.text,
      top_k: request.top_k ?? 5,
    }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed: ${res.statusText}`);
  }
  return res.json();
}

export async function generateText(request: GenerateRequest): Promise<GenerateResponse> {
  const base = getApiBaseUrl();
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
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Generation failed: ${res.statusText}`);
  }
  return res.json();
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
