"use client";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { classNames } from "@/lib/utils";
import type { GenerationStrategy } from "@/types/api";

const EXAMPLES = [
  "Once upon a time",
  "The president said that",
  "In the beginning of the story",
  "She opened the door and",
];

const NUM_WORDS_OPTIONS = [5, 10, 20, 50] as const;
const STRATEGY_OPTIONS: ReadonlyArray<{ value: GenerationStrategy; label: string }> = [
  { value: "greedy", label: "Greedy" },
  { value: "top_k", label: "Top-k" },
];

interface GenerationPanelProps {
  text: string;
  onTextChange: (next: string) => void;
  onClear: () => void;
  onGenerate: () => void;
  loading: boolean;
  disabled: boolean;
  numWords: number;
  onNumWordsChange: (next: number) => void;
  temperature: number;
  onTemperatureChange: (next: number) => void;
  strategy: GenerationStrategy;
  onStrategyChange: (next: GenerationStrategy) => void;
}

export function GenerationPanel({
  text,
  onTextChange,
  onClear,
  onGenerate,
  loading,
  disabled,
  numWords,
  onNumWordsChange,
  temperature,
  onTemperatureChange,
  strategy,
  onStrategyChange,
}: GenerationPanelProps) {
  const trimmed = text.trim();
  const canGenerate = trimmed.length > 0 && !loading && !disabled;

  return (
    <Card aria-labelledby="generation-panel-heading">
      <h2 id="generation-panel-heading" className="text-lg font-semibold text-ink">
        Text Generation
      </h2>
      <p className="mt-1 text-sm text-ink-muted">
        Autoregressively extend a prompt using the trained LSTM.
      </p>

      <label htmlFor="generate-prompt" className="sr-only">
        Prompt to extend
      </label>
      <textarea
        id="generate-prompt"
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        rows={3}
        placeholder="Once upon a time, there was a..."
        className="mt-4 w-full resize-y rounded-xl border border-surface-border bg-surface-muted p-4 font-mono text-sm text-ink outline-none transition focus:border-brand-400 focus:bg-surface-card focus:ring-2 focus:ring-brand-300 dark:focus:ring-brand-500"
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

      <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2">
        <div>
          <label
            htmlFor="num-words-slider"
            className="mb-1 block text-xs font-semibold uppercase tracking-wide text-ink-muted"
          >
            Words to generate: {numWords}
          </label>
          <input
            id="num-words-slider"
            type="range"
            min={1}
            max={50}
            step={1}
            value={numWords}
            onChange={(event) => onNumWordsChange(Number(event.target.value))}
            className="w-full accent-brand-600 dark:accent-brand-500"
          />
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            {NUM_WORDS_OPTIONS.map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => onNumWordsChange(n)}
                className={classNames(
                  "rounded-md border px-2 py-0.5 text-xs transition",
                  numWords === n
                    ? "border-brand-400 bg-brand-50 text-brand-700 dark:border-brand-500 dark:bg-brand-900/30 dark:text-brand-200"
                    : "border-surface-border bg-surface-card text-ink-muted hover:border-ink-subtle",
                )}
                aria-pressed={numWords === n}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label
            htmlFor="temperature-slider"
            className="mb-1 block text-xs font-semibold uppercase tracking-wide text-ink-muted"
          >
            Temperature: {temperature.toFixed(2)}
          </label>
          <input
            id="temperature-slider"
            type="range"
            min={0.1}
            max={2.0}
            step={0.05}
            value={temperature}
            onChange={(event) => onTemperatureChange(Number(event.target.value))}
            className="w-full accent-brand-600 dark:accent-brand-500"
          />
          <p className="mt-1 text-xs text-ink-subtle">
            Lower = deterministic. Higher = more varied.
          </p>
        </div>
      </div>

      <div className="mt-5">
        <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-ink-muted">
          Sampling strategy
        </span>
        <div className="inline-flex rounded-lg border border-surface-border bg-surface-muted p-1 text-sm">
          {STRATEGY_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => onStrategyChange(opt.value)}
              aria-pressed={strategy === opt.value}
              className={classNames(
                "rounded-md px-3 py-1.5 transition",
                strategy === opt.value
                  ? "bg-surface-card text-brand-700 shadow-sm dark:text-brand-300"
                  : "text-ink-muted hover:text-ink",
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
        <Button type="button" variant="secondary" onClick={onClear} disabled={loading || !text}>
          Clear
        </Button>
        <Button
          type="button"
          onClick={onGenerate}
          loading={loading}
          disabled={!canGenerate}
        >
          {loading ? "Generating…" : "Generate Text"}
        </Button>
      </div>

      {disabled ? (
        <p className="mt-3 text-xs text-rose-500" role="alert">
          Backend is not reachable. Start the FastAPI server to enable generation.
        </p>
      ) : null}
    </Card>
  );
}
