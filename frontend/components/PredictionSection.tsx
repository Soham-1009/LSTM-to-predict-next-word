"use client";

import React, { useState } from "react";
import { predictNextWord, CandidateToken } from "@/lib/api";

interface PredictionSectionProps {
  initialText?: string;
  onSuccess?: (prompt: string, result: string) => void;
}

export default function PredictionSection({
  initialText,
  onSuccess,
}: PredictionSectionProps) {
  const [text, setText] = useState(initialText || "Once upon a");
  const [prevInitialText, setPrevInitialText] = useState(initialText);
  if (initialText !== undefined && initialText !== prevInitialText) {
    setPrevInitialText(initialText);
    setText(initialText);
  }
  const [candidates, setCandidates] = useState<CandidateToken[]>([]);
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (inputText: string = text) => {
    if (!inputText.trim()) {
      setError("Please enter some text.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await predictNextWord({ text: inputText, top_k: topK });
      setCandidates(response.predictions);
      if (onSuccess && response.predictions.length > 0) {
        const topTokens = response.predictions
          .map((c) => `${c.token} (${(c.probability * 100).toFixed(1)}%)`)
          .join(", ");
        onSuccess(inputText, `Top predictions: [${topTokens}]`);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Prediction failed";
      if (msg.includes("Failed to fetch") || msg.includes("NetworkError") || msg.includes("fetch")) {
        setError("Backend not connected. Start FastAPI locally ('uvicorn backend.app.main:app') or click the badge in the top right to configure your backend URL.");
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSelectCandidate = (token: string) => {
    const trimmed = text.trimEnd();
    const needsSpace =
      !token.match(/^[.,!?;:'")\]}]/) &&
      !trimmed.endsWith("(") &&
      !trimmed.endsWith("[") &&
      !trimmed.endsWith("{");
    const newText = !trimmed ? token : needsSpace ? `${trimmed} ${token}` : `${trimmed}${token}`;
    setText(newText);
    handlePredict(newText);
  };

  return (
    <div className="bg-white/85 dark:bg-neutral-900/70 border border-slate-200/90 dark:border-neutral-800/80 backdrop-blur-md rounded-2xl p-6 shadow-xl space-y-5 transition-colors duration-200">
      <div className="flex flex-col sm:row sm:items-center justify-between gap-3 border-b border-slate-200 dark:border-neutral-800/60 pb-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900 dark:text-neutral-100 flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-cyan-400" />
            Interactive Next-Word Prediction & Autocomplete
          </h2>
          <p className="text-xs text-slate-500 dark:text-neutral-400 mt-0.5">
            Predict the most probable continuation token. Click any candidate suggestion to append and continue.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-600 dark:text-neutral-400 font-medium">Top-K Suggestions:</label>
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="bg-slate-100 dark:bg-neutral-800 border border-slate-300 dark:border-neutral-700 text-xs text-slate-800 dark:text-neutral-200 rounded-md px-2 py-1 focus:outline-none focus:ring-1 focus:ring-cyan-500"
          >
            <option value={3}>3</option>
            <option value={5}>5</option>
            <option value={8}>8</option>
            <option value={10}>10</option>
          </select>
        </div>
      </div>

      <div className="space-y-3">
        <textarea
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            if (error) setError(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handlePredict(text);
            }
          }}
          rows={3}
          placeholder="Type a prefix prompt here (e.g. 'machine learning', 'in the deep')..."
          className="w-full bg-slate-50 dark:bg-neutral-950/80 border border-slate-300 dark:border-neutral-800 rounded-xl p-3.5 text-slate-900 dark:text-neutral-100 placeholder-slate-400 dark:placeholder-neutral-500 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:bg-white dark:focus:bg-neutral-950/80 transition-all resize-none"
        />

        <div className="flex items-center justify-between">
          <button
            onClick={() => handlePredict()}
            disabled={loading || !text.trim()}
            className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white dark:text-neutral-950 font-medium text-xs rounded-lg shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-3.5 w-3.5 text-white dark:text-neutral-950" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Predicting...
              </>
            ) : (
              <>
                <span>Predict Next Word</span>
                <span className="font-mono text-[10px] bg-cyan-900/40 text-neutral-900 px-1.5 py-0.2 rounded font-bold">
                  ↵
                </span>
              </>
            )}
          </button>

          {candidates.length > 0 && (
            <button
              onClick={() => setCandidates([])}
              className="text-xs text-slate-500 dark:text-neutral-500 hover:text-slate-800 dark:hover:text-neutral-300 transition-colors"
            >
              Clear Suggestions
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/60 text-red-700 dark:text-red-300 text-xs rounded-lg p-3">
          {error}
        </div>
      )}

      {candidates.length > 0 && (
        <div className="space-y-2 pt-2 border-t border-slate-200 dark:border-neutral-800/60">
          <div className="text-xs font-semibold text-slate-500 dark:text-neutral-400 uppercase tracking-wider">
            Candidate Next Words:
          </div>
          <div className="flex flex-wrap gap-2">
            {candidates.map((cand, idx) => (
              <button
                key={`${cand.token}-${idx}`}
                disabled={loading}
                onClick={() => handleSelectCandidate(cand.token)}
                className="group flex items-center gap-2 bg-slate-100 dark:bg-neutral-800/90 hover:bg-cyan-50 dark:hover:bg-cyan-950/60 border border-slate-200 dark:border-neutral-700 hover:border-cyan-400 dark:hover:border-cyan-500/60 rounded-xl px-3.5 py-2 text-sm text-slate-800 dark:text-neutral-200 transition-all hover:scale-[1.02] shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                title={`Click to append "${cand.token}"`}
              >
                <span className="font-mono font-medium text-cyan-600 dark:text-cyan-300 group-hover:text-cyan-700 dark:group-hover:text-cyan-200">
                  {cand.token}
                </span>
                <span className="text-[10px] font-mono text-slate-600 dark:text-neutral-400 group-hover:text-slate-800 dark:group-hover:text-neutral-300 bg-white/90 dark:bg-neutral-900/80 px-1.5 py-0.5 rounded-full border border-slate-200 dark:border-neutral-700/60">
                  {(cand.probability * 100).toFixed(1)}%
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
