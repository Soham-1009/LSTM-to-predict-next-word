"use client";

import { Card } from "@/components/ui/Card";
import { formatPercent } from "@/lib/utils";
import type { PredictResponse, WordPrediction } from "@/types/api";

interface PredictionResultProps {
  data: PredictResponse | null;
  loading: boolean;
  error: string | null;
}

function ProbabilityBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div
      className="h-2 w-full overflow-hidden rounded-full bg-surface-muted"
      aria-hidden="true"
    >
      <div
        className="h-full rounded-full bg-gradient-to-r from-brand-400 to-brand-600 transition-all duration-500 dark:from-brand-500 dark:to-brand-300"
        style={{ width: `${pct * 100}%` }}
      />
    </div>
  );
}

function Row({ prediction }: { prediction: WordPrediction }) {
  return (
    <li className="flex items-center gap-3">
      <span className="w-28 truncate font-mono text-sm text-ink sm:w-32">
        {prediction.word}
      </span>
      <div className="flex-1">
        <ProbabilityBar value={prediction.probability} />
      </div>
      <span className="w-16 text-right font-mono text-sm tabular-nums text-ink-muted">
        {formatPercent(prediction.probability)}
      </span>
    </li>
  );
}

export function PredictionResult({ data, loading, error }: PredictionResultProps) {
  if (error) {
    return (
      <Card
        className="border-rose-200 bg-rose-50 dark:border-rose-500/40 dark:bg-rose-950/40"
        aria-live="polite"
      >
        <h2 className="text-lg font-semibold text-rose-700 dark:text-rose-300">
          Prediction failed
        </h2>
        <p className="mt-2 text-sm text-rose-700 dark:text-rose-200">{error}</p>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card aria-live="polite">
        <h2 className="text-lg font-semibold text-ink">Prediction</h2>
        <div className="mt-6 flex items-center gap-3 text-ink-muted">
          <span className="h-5 w-5 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
          <span>Querying the LSTM…</span>
        </div>
      </Card>
    );
  }

  if (!data || data.predictions.length === 0) {
    return (
      <Card className="border-dashed text-sm text-ink-muted">
        Enter a sentence and click{" "}
        <strong className="text-ink">Predict Next Word</strong> to see what the LSTM thinks comes next.
      </Card>
    );
  }

  const top = data.predictions[0];
  if (!top) return null;

  return (
    <Card aria-live="polite" className="animate-fade-in">
      <div className="flex flex-col gap-1">
        <span className="text-xs font-semibold uppercase tracking-wide text-ink-subtle">
          Input
        </span>
        <p className="font-mono text-sm text-ink-muted">{data.input_text}</p>
      </div>

      <div className="mt-5 rounded-xl bg-gradient-to-br from-brand-50 to-surface-card p-5 ring-1 ring-brand-100 dark:from-brand-900/40 dark:to-surface-card dark:ring-brand-800/60">
        <span className="text-xs font-semibold uppercase tracking-wide text-brand-600 dark:text-brand-300">
          Predicted Next Word
        </span>
        <p className="mt-1 text-3xl font-bold text-ink sm:text-4xl">{top.word}</p>
        <p className="mt-1 text-sm text-ink-muted">
          Confidence:{" "}
          <span className="font-semibold text-brand-700 dark:text-brand-300">
            {formatPercent(top.probability)}
          </span>
        </p>
      </div>

      <div className="mt-6">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-ink-muted">
          Top Predictions
        </h3>
        <ul className="space-y-3">
          {data.predictions.map((p, idx) => (
            <Row key={`${p.word}-${idx}`} prediction={p} />
          ))}
        </ul>
      </div>
    </Card>
  );
}
