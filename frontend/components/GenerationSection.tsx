"use client";

import React, { useState } from "react";
import { generateText, GenerateResponse } from "@/lib/api";

interface GenerationSectionProps {
  initialPrompt?: string;
  onSuccess?: (prompt: string, result: string) => void;
}

export default function GenerationSection({
  initialPrompt,
  onSuccess,
}: GenerationSectionProps) {
  const [prompt, setPrompt] = useState(initialPrompt || "Artificial intelligence");
  const [prevInitialPrompt, setPrevInitialPrompt] = useState(initialPrompt);
  if (initialPrompt !== prevInitialPrompt) {
    setPrevInitialPrompt(initialPrompt);
    if (initialPrompt) {
      setPrompt(initialPrompt);
    }
  }
  const [maxTokens, setMaxTokens] = useState(30);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.9);
  const [greedy, setGreedy] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError("Please provide a prompt.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await generateText({
        prompt,
        max_new_tokens: maxTokens,
        temperature: greedy ? 0.0 : temperature,
        top_k: topK,
        top_p: topP,
        greedy,
      });
      setResult(response);
      if (onSuccess) {
        onSuccess(prompt, response.generated_text);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Generation failed";
      if (msg.includes("Failed to fetch") || msg.includes("NetworkError") || msg.includes("fetch")) {
        setError("Backend not connected. Start FastAPI locally ('uvicorn backend.app.main:app') or configure your URL in the status badge.");
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-5 space-y-4">
      <div className="pb-3 border-b border-[var(--rule)]">
        <h2 className="text-base font-semibold text-[var(--text)]">
          Autoregressive Sequence Generation
        </h2>
        <p className="text-[13px] text-[var(--muted)] mt-0.5">
          Iterative autoregressive token generation using nucleus sampling, temperature scaling, and greedy decoding.
        </p>
      </div>

      <div className="space-y-4">
        <div className="space-y-1.5">
          <label className="text-[13px] font-medium text-[var(--muted)]">Seed prompt:</label>
          <textarea
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value);
              if (error) setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleGenerate();
              }
            }}
            rows={3}
            placeholder="Enter seed prompt sequence..."
            className="w-full bg-[var(--ground)] border border-[var(--control)] rounded p-3 text-sm text-[var(--text)] placeholder-[var(--faint)] font-mono resize-none focus:outline-none focus:border-[var(--accent)] leading-relaxed"
          />
        </div>

        {/* Hyperparameters / Sampling Controls Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 p-3.5 bg-[var(--ground)] border border-[var(--border)] rounded">
          <div className="space-y-1">
            <div className="flex justify-between text-[13px]">
              <span className="text-[var(--muted)]">Temperature</span>
              <span className="tabular-nums text-[var(--text)] font-medium">
                {greedy ? "0.0 (Greedy)" : temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.05"
              value={temperature}
              disabled={greedy}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full cursor-pointer disabled:opacity-40"
              style={{ accentColor: "var(--accent)" }}
            />
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-[13px]">
              <span className="text-[var(--muted)]">Top-K</span>
              <span className="tabular-nums text-[var(--text)] font-medium">
                {greedy ? "Overridden" : topK}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={topK}
              disabled={greedy}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full cursor-pointer disabled:opacity-40"
              style={{ accentColor: "var(--accent)" }}
            />
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-[13px]">
              <span className="text-[var(--muted)]">Top-P (Nucleus)</span>
              <span className="tabular-nums text-[var(--text)] font-medium">
                {greedy ? "Overridden" : topP.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={topP}
              disabled={greedy}
              onChange={(e) => setTopP(parseFloat(e.target.value))}
              className="w-full cursor-pointer disabled:opacity-40"
              style={{ accentColor: "var(--accent)" }}
            />
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-[13px]">
              <span className="text-[var(--muted)]">Max Tokens</span>
              <span className="tabular-nums text-[var(--text)] font-medium">
                {maxTokens}
              </span>
            </div>
            <input
              type="range"
              min="5"
              max="100"
              step="5"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full cursor-pointer"
              style={{ accentColor: "var(--accent)" }}
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <label className="flex items-center gap-2 cursor-pointer select-none text-[13px] text-[var(--text)]">
            <input
              type="checkbox"
              checked={greedy}
              onChange={(e) => setGreedy(e.target.checked)}
              className="rounded border-[var(--control)] text-[var(--accent)] focus:ring-0"
              style={{ accentColor: "var(--accent)" }}
            />
            <span>Greedy decoding (deterministic argmax)</span>
          </label>

          <button
            onClick={handleGenerate}
            disabled={loading || !prompt.trim()}
            className="px-4 py-2 bg-[var(--accent)] hover:opacity-90 text-white font-medium text-[13px] rounded transition-opacity duration-150 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 cursor-pointer"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-3.5 w-3.5 text-white" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <span>Generating...</span>
              </>
            ) : (
              <span>Generate sequence</span>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-[var(--pill-negative-bg)] border border-[var(--negative)] text-[var(--pill-negative-ink)] text-[13px] rounded p-3">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-2 pt-3 border-t border-[var(--rule)]">
          <div className="flex items-center justify-between text-[13px] text-[var(--muted)]">
            <span className="font-medium text-[var(--text)]">Generated text continuation:</span>
            <span className="tabular-nums text-xs text-[var(--muted)]">
              {result.tokens_generated} tokens produced · stop: {result.finish_reason}
            </span>
          </div>
          <div className="bg-[var(--ground)] border border-[var(--border)] rounded p-4 font-mono text-[13px] text-[var(--text)] whitespace-pre-wrap leading-relaxed">
            <span className="text-[var(--accent)] font-semibold">{result.prompt}</span>
            <span>
              {(() => {
                const tokenRegex = /\w+(?:'\w+)?|[^\w\s]/g;
                const promptTokens = result.prompt.toLowerCase().match(tokenRegex) || [];
                if (promptTokens.length === 0) {
                  return result.generated_text ? (result.generated_text.startsWith(" ") ? result.generated_text : ` ${result.generated_text}`) : "";
                }
                const genText = result.generated_text;
                let count = 0;
                let splitIndex = 0;
                while (tokenRegex.exec(genText) !== null) {
                  count++;
                  if (count === promptTokens.length) {
                    splitIndex = tokenRegex.lastIndex;
                    break;
                  }
                }
                if (count < promptTokens.length) return "";
                const rem = genText.slice(splitIndex);
                if (!rem) return "";
                if (!rem.startsWith(" ") && !/^[.,!?;:'")\]}]/.test(rem)) {
                  return ` ${rem}`;
                }
                return rem;
              })()}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
