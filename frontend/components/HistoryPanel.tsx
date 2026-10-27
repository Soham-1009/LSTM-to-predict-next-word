"use client";

import React from "react";

export interface HistoryItem {
  id: string;
  type: "prediction" | "generation";
  prompt: string;
  result: string;
  timestamp: string;
}

interface HistoryPanelProps {
  history: HistoryItem[];
  onSelectPrompt: (prompt: string, type: "prediction" | "generation") => void;
  onClearHistory: () => void;
}

export default function HistoryPanel({
  history,
  onSelectPrompt,
  onClearHistory,
}: HistoryPanelProps) {
  if (history.length === 0) {
    return (
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-5 text-center">
        <p className="text-[13px] text-[var(--muted)]">
          No inference history in this session yet. Perform an autocomplete prediction or sequence generation to view records.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-5 space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-[var(--rule)]">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-[var(--text)]">
            Session History
          </h2>
          <span className="tabular-nums text-xs px-2 py-0.5 rounded bg-[var(--raised)] border border-[var(--border)] text-[var(--muted)]">
            {history.length}
          </span>
        </div>
        <button
          onClick={onClearHistory}
          className="text-[13px] text-[var(--muted)] hover:text-[var(--negative)] transition-colors duration-150 cursor-pointer"
        >
          Clear history
        </button>
      </div>

      <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
        {history.map((item) => (
          <div
            key={item.id}
            className="p-3 bg-[var(--ground)] border border-[var(--border)] rounded space-y-1.5 text-[13px]"
          >
            <div className="flex items-center justify-between">
              <span className="tabular-nums text-xs font-medium text-[var(--accent)]">
                {item.type === "prediction" ? "Autocomplete" : "Generation"}
              </span>
              <span className="tabular-nums text-xs text-[var(--faint)]">
                {item.timestamp}
              </span>
            </div>

            <div className="flex items-baseline justify-between gap-2">
              <div className="text-[var(--text)] truncate">
                <span className="text-[var(--muted)]">Prompt: </span>
                <span className="font-mono text-[13px]">&quot;{item.prompt}&quot;</span>
              </div>
              <button
                onClick={() => onSelectPrompt(item.prompt, item.type)}
                className="text-xs text-[var(--accent)] hover:underline shrink-0 font-medium cursor-pointer"
              >
                Reuse prompt
              </button>
            </div>

            <div className="font-mono text-xs text-[var(--muted)] bg-[var(--surface)] border border-[var(--border)] p-2 rounded break-words leading-relaxed">
              <span className="text-[var(--faint)]">Result: </span>
              {item.result}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
