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
      <div className="bg-white/70 dark:bg-neutral-900/40 border border-slate-200/90 dark:border-neutral-800/60 rounded-2xl p-6 text-center shadow-sm transition-colors duration-200">
        <div className="text-xs text-slate-500 dark:text-neutral-500 font-mono">
          No inference history in this session yet. Perform a prediction or generate text to see records here.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/85 dark:bg-neutral-900/60 border border-slate-200/90 dark:border-neutral-800/80 rounded-2xl p-6 shadow-xl space-y-4 transition-colors duration-200">
      <div className="flex items-center justify-between border-b border-slate-200 dark:border-neutral-800/60 pb-3">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
          <h2 className="text-base font-semibold text-slate-900 dark:text-neutral-100">Session History</h2>
          <span className="text-xs font-mono bg-slate-200 dark:bg-neutral-800 text-slate-700 dark:text-neutral-400 px-2 py-0.5 rounded-full">
            {history.length}
          </span>
        </div>
        <button
          onClick={onClearHistory}
          className="text-xs text-slate-500 dark:text-neutral-500 hover:text-red-500 dark:hover:text-red-400 transition-colors"
        >
          Clear History
        </button>
      </div>

      <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
        {history.map((item) => (
          <div
            key={item.id}
            className="p-3 bg-slate-50 dark:bg-neutral-950/70 border border-slate-200 dark:border-neutral-800/70 rounded-xl space-y-1.5 hover:border-slate-300 dark:hover:border-neutral-700 transition-all text-xs"
          >
            <div className="flex items-center justify-between">
              <span
                className={`font-mono text-[10px] uppercase px-1.5 py-0.5 rounded font-medium ${
                  item.type === "prediction"
                    ? "bg-cyan-100 dark:bg-cyan-950 text-cyan-800 dark:text-cyan-300 border border-cyan-200 dark:border-cyan-800/60"
                    : "bg-violet-100 dark:bg-violet-950 text-violet-800 dark:text-violet-300 border border-violet-200 dark:border-violet-800/60"
                }`}
              >
                {item.type}
              </span>
              <span className="text-[10px] font-mono text-slate-500 dark:text-neutral-500">{item.timestamp}</span>
            </div>

            <div className="flex items-baseline justify-between gap-2">
              <div className="font-mono text-slate-700 dark:text-neutral-300 truncate">
                <span className="text-slate-500 dark:text-neutral-500">Prompt: </span>
                <span className="text-slate-900 dark:text-neutral-200">&quot;{item.prompt}&quot;</span>
              </div>
              <button
                onClick={() => onSelectPrompt(item.prompt, item.type)}
                className="text-[11px] text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 shrink-0 font-medium"
              >
                Reuse Prompt ↵
              </button>
            </div>

            <div className="font-mono text-slate-700 dark:text-neutral-400 bg-white dark:bg-neutral-900/60 border border-slate-200 dark:border-transparent p-2 rounded-lg break-words">
              <span className="text-slate-500 dark:text-neutral-500">Output: </span>
              {item.result}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
