"use client";

import type { BackendStatus } from "@/hooks/useBackendStatus";

interface HeaderProps {
  backendStatus: BackendStatus;
}

const DOT_CLASS: Record<BackendStatus["state"], string> = {
  online: "bg-emerald-500",
  degraded: "bg-amber-500",
  offline: "bg-rose-500",
  checking: "bg-slate-400 dark:bg-slate-500",
};

const LABEL: Record<BackendStatus["state"], string> = {
  online: "Backend connected",
  degraded: "Model not loaded",
  offline: "Backend offline",
  checking: "Checking…",
};

export function Header({ backendStatus }: HeaderProps) {
  const { state, data, error } = backendStatus;
  const tooltip = error || (data ? `Device: ${data.device}` : "");

  return (
    <header className="border-b border-surface-border bg-surface-card/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-6 py-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
            LSTM Next Word Predictor
          </h1>
          <p className="text-sm text-ink-muted">
            Generative AI Project — Next Word Prediction &amp; Text Generation using LSTM
          </p>
        </div>
        <div
          className="inline-flex items-center gap-2 self-start rounded-full border border-surface-border bg-surface-card px-3 py-1.5 text-sm text-ink-muted shadow-soft sm:self-auto"
          title={tooltip}
        >
          <span
            className={`h-2.5 w-2.5 rounded-full ${DOT_CLASS[state]}`}
            aria-hidden="true"
          />
          <span className="font-medium">{LABEL[state]}</span>
          {data?.device ? (
            <span className="rounded bg-surface-muted px-1.5 py-0.5 text-xs uppercase tracking-wide text-ink-muted">
              {data.device}
            </span>
          ) : null}
        </div>
      </div>
    </header>
  );
}
