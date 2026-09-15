"use client";

import { useState } from "react";
import { GenerationPanel } from "@/components/generation/GenerationPanel";
import { GenerationResult } from "@/components/generation/GenerationResult";
import { Header } from "@/components/dashboard/Header";
import { HistoryPanel } from "@/components/history/HistoryPanel";
import { HowItWorksPanel } from "@/components/education/HowItWorksPanel";
import { ModelInfoPanel } from "@/components/model/ModelInfoPanel";
import { PredictionPanel } from "@/components/prediction/PredictionPanel";
import { PredictionResult } from "@/components/prediction/PredictionResult";
import { useBackendStatus } from "@/hooks/useBackendStatus";
import { useLocalHistory } from "@/hooks/useLocalHistory";
import { usePrediction } from "@/hooks/usePrediction";
import { classNames } from "@/lib/utils";
import type { GenerationStrategy, HistoryEntry } from "@/types/api";

const DEFAULT_TEMPERATURE = 1.0;

export function Dashboard() {
  const backendStatus = useBackendStatus();
  const { state, predict, generate, reset } = usePrediction();
  const { history, addEntry, removeEntry, clear } = useLocalHistory();

  const [mode, setMode] = useState<"single" | "multi">("single");

  const [singleText, setSingleText] = useState("");
  const [topK, setTopK] = useState(5);

  const [multiText, setMultiText] = useState("");
  const [numWords, setNumWords] = useState(10);
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [strategy, setStrategy] = useState<GenerationStrategy>("greedy");

  const isOffline = backendStatus.state === "offline";
  const loading = state.status === "loading";

  async function handlePredict(): Promise<void> {
    try {
      const result = await predict({ text: singleText.trim(), topK });
      if (!result) return;
      const top = result.predictions[0];
      if (top) {
        addEntry({
          kind: "predict",
          input: result.input_text,
          output: top.word,
          confidence: top.probability,
        });
      }
    } catch {
      /* error already in state */
    }
  }

  async function handleGenerate(): Promise<void> {
    try {
      const result = await generate({
        text: multiText.trim(),
        numWords,
        temperature,
        strategy,
      });
      if (!result) return;
      addEntry({
        kind: "generate",
        input: result.input_text,
        output: result.generated_text,
        generated_words: result.generated_words,
      });
    } catch {
      /* error already in state */
    }
  }

  function clearSingle(): void {
    setSingleText("");
    if (mode === "single") reset();
  }

  function clearMulti(): void {
    setMultiText("");
    if (mode === "multi") reset();
  }

  function pickFromHistory(entry: HistoryEntry): void {
    if (entry.kind === "predict") {
      setMode("single");
      setSingleText(entry.input);
    } else {
      setMode("multi");
      setMultiText(entry.input);
    }
  }

  return (
    <div className="min-h-screen bg-surface-page text-ink">
      <Header backendStatus={backendStatus} />

      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <div
          role="tablist"
          aria-label="Dashboard mode"
          className="mb-6 inline-flex rounded-lg border border-surface-border bg-surface-card p-1 text-sm shadow-soft"
        >
          {(
            [
              { id: "single", label: "Predict next word" },
              { id: "multi", label: "Generate text" },
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={mode === tab.id}
              onClick={() => setMode(tab.id)}
              className={classNames(
                "rounded-md px-4 py-1.5 transition",
                mode === tab.id
                  ? "bg-brand-600 font-semibold text-white shadow-sm"
                  : "text-ink-muted hover:text-ink",
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
            {mode === "single" ? (
              <>
                <PredictionPanel
                  text={singleText}
                  onTextChange={setSingleText}
                  onClear={clearSingle}
                  onSubmit={() => void handlePredict()}
                  loading={loading}
                  disabled={isOffline}
                  topK={topK}
                  onTopKChange={setTopK}
                />
                <PredictionResult
                  data={state.predictions}
                  loading={loading}
                  error={state.status === "error" ? state.error : null}
                />
              </>
            ) : (
              <>
                <GenerationPanel
                  text={multiText}
                  onTextChange={setMultiText}
                  onClear={clearMulti}
                  onGenerate={() => void handleGenerate()}
                  loading={loading}
                  disabled={isOffline}
                  numWords={numWords}
                  onNumWordsChange={setNumWords}
                  temperature={temperature}
                  onTemperatureChange={setTemperature}
                  strategy={strategy}
                  onStrategyChange={setStrategy}
                />
                <GenerationResult
                  data={state.generated}
                  loading={loading}
                  error={state.status === "error" ? state.error : null}
                />
              </>
            )}

            <HowItWorksPanel />
          </div>

          <aside className="space-y-6">
            <ModelInfoPanel />
            <HistoryPanel
              history={history}
              onPick={pickFromHistory}
              onRemove={removeEntry}
              onClear={clear}
            />
          </aside>
        </div>

        <footer className="mt-12 text-center text-xs text-ink-subtle">
          Built with Next.js, TypeScript, Tailwind CSS, FastAPI, and PyTorch · LSTM Next Word
          Predictor · {new Date().getFullYear()}
        </footer>
      </main>
    </div>
  );
}
