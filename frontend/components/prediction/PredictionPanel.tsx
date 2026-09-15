"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { classNames } from "@/lib/utils";

const EXAMPLES = [
  "Artificial intelligence is",
  "Machine learning can",
  "The future of technology",
  "Deep learning is",
  "Natural language processing helps",
];

const TOP_K_OPTIONS = [1, 3, 5, 10] as const;

interface PredictionPanelProps {
  text: string;
  onTextChange: (next: string) => void;
  onClear: () => void;
  onSubmit: () => void;
  loading: boolean;
  disabled: boolean;
  topK: number;
  onTopKChange: (next: number) => void;
}

export function PredictionPanel({
  text,
  onTextChange,
  onClear,
  onSubmit,
  loading,
  disabled,
  topK,
  onTopKChange,
}: PredictionPanelProps) {
  const [copied, setCopied] = useState(false);
  const trimmed = text.trim();
  const canPredict = trimmed.length > 0 && !loading && !disabled;
  const characterCount = text.length;

  async function copy(): Promise<void> {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard may be unavailable */
    }
  }

  return (
    <Card aria-labelledby="prediction-panel-heading">
      <div className="mb-4 flex items-center justify-between">
        <h2 id="prediction-panel-heading" className="text-lg font-semibold text-ink">
          Input Text
        </h2>
        <span className="text-xs font-medium text-ink-subtle">
          {characterCount} character{characterCount === 1 ? "" : "s"}
        </span>
      </div>

      <label htmlFor="prompt-textarea" className="sr-only">
        Enter a sentence to predict the next word
      </label>
      <textarea
        id="prompt-textarea"
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        placeholder="Type a sentence here...&#10;Example: Artificial intelligence is"
        rows={4}
        className="w-full resize-y rounded-xl border border-surface-border bg-surface-muted p-4 font-mono text-sm text-ink outline-none transition focus:border-brand-400 focus:bg-surface-card focus:ring-2 focus:ring-brand-300 dark:focus:ring-brand-500"
      />

      <div className="mt-4 flex flex-wrap items-center gap-2">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onTextChange(example)}
            className="rounded-full border border-surface-border bg-surface-card px-3 py-1 text-xs text-ink-muted transition hover:border-brand-400 hover:bg-brand-50 hover:text-brand-700 dark:hover:bg-brand-900/30 dark:hover:text-brand-200"
          >
            {example}
          </button>
        ))}
      </div>

      <div className="mt-6">
        <span className="mb-2 block text-xs font-semibold uppercase tracking-wide text-ink-muted">
          Number of Predictions
        </span>
        <div className="inline-flex rounded-lg border border-surface-border bg-surface-muted p-1 text-sm">
          {TOP_K_OPTIONS.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => onTopKChange(n)}
              aria-pressed={topK === n}
              className={classNames(
                "rounded-md px-3 py-1.5 transition",
                topK === n
                  ? "bg-surface-card text-brand-700 shadow-sm dark:bg-surface-card dark:text-brand-300"
                  : "text-ink-muted hover:text-ink",
              )}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            variant="secondary"
            onClick={onClear}
            disabled={loading || !text}
          >
            Clear
          </Button>
          <Button
            type="button"
            variant="secondary"
            onClick={() => void copy()}
            disabled={!text}
          >
            {copied ? "Copied!" : "Copy"}
          </Button>
        </div>

        <Button
          type="button"
          onClick={onSubmit}
          loading={loading}
          disabled={!canPredict}
          aria-label="Predict next word"
        >
          {loading ? "Predicting…" : "Predict Next Word"}
        </Button>
      </div>

      {disabled ? (
        <p className="mt-3 text-xs text-rose-500" role="alert">
          Backend is not reachable. Start the FastAPI server to enable predictions.
        </p>
      ) : null}
    </Card>
  );
}
