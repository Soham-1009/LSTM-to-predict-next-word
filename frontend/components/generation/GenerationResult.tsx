"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import type { GenerateResponse } from "@/types/api";

interface GenerationResultProps {
  data: GenerateResponse | null;
  loading: boolean;
  error: string | null;
}

export function GenerationResult({ data, loading, error }: GenerationResultProps) {
  const [copied, setCopied] = useState(false);

  async function copy(): Promise<void> {
    if (!data?.generated_text) return;
    try {
      await navigator.clipboard.writeText(data.generated_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  }

  if (error) {
    return (
      <Card
        className="border-rose-200 bg-rose-50 dark:border-rose-500/40 dark:bg-rose-950/40"
        aria-live="polite"
      >
        <h2 className="text-lg font-semibold text-rose-700 dark:text-rose-300">
          Generation failed
        </h2>
        <p className="mt-2 text-sm text-rose-700 dark:text-rose-200">{error}</p>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card aria-live="polite">
        <h2 className="text-lg font-semibold text-ink">Generating…</h2>
        <p className="mt-2 text-sm text-ink-muted">
          The LSTM is producing words one at a time.
        </p>
        <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-surface-muted">
          <div className="h-full w-1/3 animate-pulse rounded-full bg-brand-500" />
        </div>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className="border-dashed text-sm text-ink-muted">
        Enter a prompt above and click <strong className="text-ink">Generate Text</strong> to
        extend it word by word.
      </Card>
    );
  }

  return (
    <Card aria-live="polite" className="animate-fade-in">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-ink">Generated Text</h2>
        <Button type="button" variant="secondary" size="sm" onClick={() => void copy()}>
          {copied ? "Copied!" : "Copy"}
        </Button>
      </div>

      <p className="rounded-xl bg-surface-muted p-4 font-mono text-sm leading-relaxed text-ink">
        {data.generated_text}
      </p>

      {data.generated_words.length > 0 ? (
        <div className="mt-4">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">
            Generated tokens
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {data.generated_words.map((word, index) => (
              <span
                key={`${word}-${index}`}
                className="rounded-full bg-brand-100 px-2.5 py-0.5 font-mono text-xs text-brand-800 dark:bg-brand-900/40 dark:text-brand-200"
              >
                {word}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </Card>
  );
}
