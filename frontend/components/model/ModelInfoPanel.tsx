"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/Card";
import { api, ApiError } from "@/lib/api";
import type { ModelInfo } from "@/types/api";

const FALLBACK: ModelInfo = {
  architecture: "LSTM",
  framework: "PyTorch",
  vocab_size: 0,
  embedding_dim: 0,
  hidden_dim: 0,
  num_layers: 0,
  dropout: 0,
  sequence_length: 0,
  device: "—",
  parameter_count: null,
};

export function ModelInfoPanel() {
  const [open, setOpen] = useState(false);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || info) return;
    const controller = new AbortController();
    api
      .modelInfo(controller.signal)
      .then((data) => setInfo(data))
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(err instanceof ApiError ? err.message : "Could not load model info.");
      });
    return () => controller.abort();
  }, [open, info]);

  const data: ModelInfo = info ?? FALLBACK;
  const rows: Array<[string, string]> = [
    ["Architecture", data.architecture],
    ["Framework", data.framework],
    ["Vocabulary size", data.vocab_size > 0 ? data.vocab_size.toLocaleString() : "—"],
    ["Embedding dim", String(data.embedding_dim || "—")],
    ["Hidden dim", String(data.hidden_dim || "—")],
    ["LSTM layers", String(data.num_layers || "—")],
    ["Dropout", String(data.dropout || "—")],
    ["Sequence length", String(data.sequence_length || "—")],
    ["Device", data.device],
  ];
  if (data.parameter_count != null) {
    rows.push(["Parameters", data.parameter_count.toLocaleString()]);
  }

  return (
    <Card as="section" aria-labelledby="model-info-heading">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="model-info-body"
        className="flex w-full items-center justify-between text-left"
      >
        <span id="model-info-heading" className="text-lg font-semibold text-ink">
          Model Information
        </span>
        <span
          className={`text-ink-subtle transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden="true"
        >
          ▾
        </span>
      </button>
      {open ? (
        <div id="model-info-body" className="mt-5 border-t border-surface-border pt-4">
          {error ? (
            <p className="text-sm text-rose-500" role="alert">
              {error}
            </p>
          ) : (
            <dl className="grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2">
              {rows.map(([label, value]) => (
                <div
                  key={label}
                  className="flex justify-between border-b border-surface-border py-1.5 text-sm last:border-0 sm:border-0"
                >
                  <dt className="text-ink-muted">{label}</dt>
                  <dd className="font-mono text-ink">{value}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      ) : null}
    </Card>
  );
}
