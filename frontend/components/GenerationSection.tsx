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
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white/85 dark:bg-neutral-900/70 border border-slate-200/90 dark:border-neutral-800/80 backdrop-blur-md rounded-2xl p-6 shadow-xl space-y-5 transition-colors duration-200">
      <div className="border-b border-slate-200 dark:border-neutral-800/60 pb-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-neutral-100 flex items-center gap-2">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-violet-400" />
          Autoregressive Text Generation
        </h2>
        <p className="text-xs text-slate-500 dark:text-neutral-400 mt-0.5">
          Generate natural text continuations using configurable sampling strategies (Greedy, Temperature, Top-K, Nucleus/Top-P).
        </p>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-xs font-medium text-slate-700 dark:text-neutral-300">Prompt:</label>
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
            placeholder="Enter seed prompt..."
            className="w-full bg-slate-50 dark:bg-neutral-950/80 border border-slate-300 dark:border-neutral-800 rounded-xl p-3.5 text-slate-900 dark:text-neutral-100 placeholder-slate-400 dark:placeholder-neutral-500 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:bg-white dark:focus:bg-neutral-950/80 transition-all resize-none"
          />
        </div>

        {/* Hyperparameters / Sampling Controls Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 p-4 bg-slate-50/80 dark:bg-neutral-950/60 border border-slate-200 dark:border-neutral-800/70 rounded-xl">
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-600 dark:text-neutral-400">Temperature</span>
              <span className="font-mono text-violet-600 dark:text-violet-400">{greedy ? "0.0 (Greedy)" : temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.05"
              value={temperature}
              disabled={greedy}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-violet-500 cursor-pointer disabled:opacity-40"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-600 dark:text-neutral-400">Top-K</span>
              <span className="font-mono text-violet-600 dark:text-violet-400">{greedy ? "Overridden" : topK}</span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={topK}
              disabled={greedy}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full accent-violet-500 cursor-pointer disabled:opacity-40"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-600 dark:text-neutral-400">Top-P (Nucleus)</span>
              <span className="font-mono text-violet-600 dark:text-violet-400">{greedy ? "Overridden" : topP.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={topP}
              disabled={greedy}
              onChange={(e) => setTopP(parseFloat(e.target.value))}
              className="w-full accent-violet-500 cursor-pointer disabled:opacity-40"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-600 dark:text-neutral-400">Max Tokens</span>
              <span className="font-mono text-violet-600 dark:text-violet-400">{maxTokens}</span>
            </div>
            <input
              type="range"
              min="5"
              max="100"
              step="5"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full accent-violet-500 cursor-pointer"
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <label className="flex items-center gap-2 cursor-pointer select-none text-xs text-slate-700 dark:text-neutral-300">
            <input
              type="checkbox"
              checked={greedy}
              onChange={(e) => setGreedy(e.target.checked)}
              className="rounded bg-slate-100 dark:bg-neutral-800 border-slate-300 dark:border-neutral-700 text-violet-500 focus:ring-violet-500/40"
            />
            <span>Greedy Decoding (Always select argmax)</span>
          </label>

          <button
            onClick={handleGenerate}
            disabled={loading || !prompt.trim()}
            className="px-4 py-2 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-medium text-xs rounded-lg shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-3.5 w-3.5 text-white" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Generating Sequence...
              </>
            ) : (
              <span>Generate Sequence</span>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/60 text-red-700 dark:text-red-300 text-xs rounded-lg p-3">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-2 pt-2 border-t border-slate-200 dark:border-neutral-800/60">
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-neutral-400">
            <span className="font-semibold uppercase tracking-wider">Generated Output:</span>
            <span className="font-mono text-[11px] text-slate-500 dark:text-neutral-500">
              {result.tokens_generated} tokens • reason: {result.finish_reason}
            </span>
          </div>
          <div className="bg-slate-50 dark:bg-neutral-950/90 border border-slate-200 dark:border-neutral-800 rounded-xl p-4 font-mono text-sm text-slate-900 dark:text-neutral-200 whitespace-pre-wrap leading-relaxed">
            <span className="text-violet-700 dark:text-violet-300 font-semibold">{result.prompt}</span>
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
