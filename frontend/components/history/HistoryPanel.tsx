"use client";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { formatTime, truncate } from "@/lib/utils";
import type { HistoryEntry } from "@/types/api";

interface HistoryPanelProps {
  history: HistoryEntry[];
  onPick: (entry: HistoryEntry) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}

export function HistoryPanel({ history, onPick, onRemove, onClear }: HistoryPanelProps) {
  if (history.length === 0) {
    return (
      <Card aria-labelledby="history-heading">
        <h2 id="history-heading" className="text-lg font-semibold text-ink">
          Recent Predictions
        </h2>
        <p className="mt-2 text-sm text-ink-muted">
          Your recent predictions will appear here. History is stored only in your browser.
        </p>
      </Card>
    );
  }

  return (
    <Card aria-labelledby="history-heading">
      <div className="flex items-center justify-between">
        <h2 id="history-heading" className="text-lg font-semibold text-ink">
          Recent Predictions
        </h2>
        <Button type="button" variant="ghost" size="sm" onClick={onClear}>
          Clear all
        </Button>
      </div>

      <ul className="mt-4 divide-y divide-surface-border">
        {history.map((entry) => (
          <li key={entry.id} className="py-3 first:pt-0 last:pb-0">
            <div className="flex items-start justify-between gap-2">
              <button
                type="button"
                onClick={() => onPick(entry)}
                className="flex-1 text-left text-ink transition hover:text-brand-700 dark:hover:text-brand-300"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-sm">
                    {truncate(entry.input, 40)}
                  </span>
                  <span className="shrink-0 text-xs text-ink-subtle">{formatTime(entry.ts)}</span>
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-ink-muted">
                  <span className="rounded bg-surface-muted px-1.5 py-0.5 font-mono text-[11px] uppercase tracking-wide text-ink-muted">
                    {entry.kind === "generate" ? "Generated" : "Top word"}
                  </span>
                  <span className="break-all font-mono text-ink-muted">
                    {truncate(entry.output, 80)}
                  </span>
                </div>
              </button>
              <button
                type="button"
                onClick={() => onRemove(entry.id)}
                aria-label="Remove from history"
                className="shrink-0 rounded p-1 text-ink-subtle transition hover:bg-rose-50 hover:text-rose-500 dark:hover:bg-rose-950/40"
              >
                <span aria-hidden="true">×</span>
              </button>
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}
