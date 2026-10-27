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
        setError("Backend not connected. Start FastAPI locally ('uvicorn backend.app.main:app') or configure your URL in the status badge.");
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
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 pb-3 border-b border-[var(--rule)]">
        <div>
          <h2 className="text-base font-semibold text-[var(--text)]">
            Next-Word Autocomplete
          </h2>
          <p className="text-[13px] text-[var(--muted)] mt-0.5">
            Predict continuation tokens conditioned on the preceding context. Click any token to append.
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <label className="text-[13px] text-[var(--muted)]">Candidates:</label>
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="bg-[var(--ground)] border border-[var(--control)] text-[13px] text-[var(--text)] rounded px-2.5 py-1 focus:outline-none focus:border-[var(--accent)]"
          >
            <option value={3}>3 tokens</option>
            <option value={5}>5 tokens</option>
            <option value={8}>8 tokens</option>
            <option value={10}>10 tokens</option>
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
          rows={4}
          placeholder="Enter a prompt sequence (e.g. 'Once upon a', 'in computer science a')..."
          className="w-full bg-[var(--ground)] border border-[var(--control)] rounded p-3 text-sm text-[var(--text)] placeholder-[var(--faint)] focus:outline-none focus:border-[var(--accent)] font-mono resize-none leading-relaxed"
        />

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-xs text-[var(--muted)]">Presets:</span>
            {[
              "Once upon a",
              "In computer science",
              "Deep neural networks",
            ].map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => {
                  setText(preset);
                  handlePredict(preset);
                }}
                className="text-xs px-2 py-0.5 rounded bg-[var(--raised)] border border-[var(--border)] hover:border-[var(--accent)] text-[var(--muted)] hover:text-[var(--text)] transition-colors cursor-pointer"
              >
                {preset}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => handlePredict()}
              disabled={loading || !text.trim()}
              className="px-4 py-2 bg-[var(--accent)] hover:opacity-90 text-white font-medium text-[13px] rounded transition-opacity duration-150 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 cursor-pointer"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-3.5 w-3.5 text-white" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>Predicting...</span>
                </>
              ) : (
                <>
                  <span>Predict next token</span>
                  <kbd className="text-xs bg-black/30 text-white px-1.5 py-0.5 rounded font-mono">
                    Enter
                  </kbd>
                </>
              )}
            </button>

            {candidates.length > 0 && (
              <button
                onClick={() => setCandidates([])}
                className="text-[13px] text-[var(--muted)] hover:text-[var(--text)] transition-colors duration-150 cursor-pointer"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-[var(--pill-negative-bg)] border border-[var(--negative)] text-[var(--pill-negative-ink)] text-[13px] rounded p-3">
          {error}
        </div>
      )}

      {candidates.length > 0 && (
        <div className="space-y-2 pt-3 border-t border-[var(--rule)]">
          <div className="text-[13px] font-medium text-[var(--muted)]">
            Candidate continuation tokens
          </div>
          <div className="flex flex-wrap gap-2">
            {candidates.map((cand, idx) => (
              <button
                key={`${cand.token}-${idx}`}
                disabled={loading}
                onClick={() => handleSelectCandidate(cand.token)}
                className="group flex items-center gap-2 bg-[var(--raised)] hover:bg-[var(--ground)] border border-[var(--border)] hover:border-[var(--accent)] rounded px-3 py-1.5 text-[13px] text-[var(--text)] transition-colors duration-150 disabled:opacity-50 cursor-pointer"
                title={`Click to append "${cand.token}"`}
              >
                <span className="font-mono font-semibold text-[var(--accent)]">
                  {cand.token}
                </span>
                <span className="tabular-nums text-xs text-[var(--muted)] group-hover:text-[var(--text)]">
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
