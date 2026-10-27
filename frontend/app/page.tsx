"use client";

import React, { useState } from "react";
import ModelInfoBadge from "@/components/ModelInfoBadge";
import ThemeToggle from "@/components/ThemeToggle";
import PredictionSection from "@/components/PredictionSection";
import GenerationSection from "@/components/GenerationSection";
import HistoryPanel, { HistoryItem } from "@/components/HistoryPanel";
import HowItWorks from "@/components/HowItWorks";
import GeneralizationSection from "@/components/GeneralizationSection";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"predict" | "generate" | "evaluation" | "how-it-works">("predict");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedPrompt, setSelectedPrompt] = useState<string>("");

  const handleAddHistory = (type: "prediction" | "generation", prompt: string, result: string) => {
    const newItem: HistoryItem = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      prompt,
      result,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    };
    setHistory((prev) => [newItem, ...prev.slice(0, 19)]);
  };

  const handleSelectHistoryPrompt = (prompt: string, type: "prediction" | "generation") => {
    setSelectedPrompt(prompt);
    setActiveTab(type === "prediction" ? "predict" : "generate");
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-[100svh] bg-[var(--ground)] text-[var(--text)] flex flex-col">
      {/* Top Header Bar */}
      <header className="border-b border-[var(--border)] bg-[var(--surface)] sticky top-0 z-40">
        <div className="max-w-6xl w-full mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded bg-[var(--raised)] border border-[var(--border)] flex items-center justify-center font-bold text-xs text-[var(--accent)] shrink-0">
              LM
            </div>
            <div className="flex items-baseline gap-2">
              <span className="font-semibold text-sm tracking-tight text-[var(--text)]">
                LSTM Language Model
              </span>
              <span className="text-xs font-mono text-[var(--faint)] hidden sm:inline">
                v0.1
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ModelInfoBadge />
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Canvas */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Title Block */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 pb-2 border-b border-[var(--rule)]">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-[var(--text)]">
              Language Model Workbench
            </h1>
            <p className="text-[13px] text-[var(--muted)] mt-1">
              Causal token prediction, autoregressive sequence generation, and multi-domain empirical evaluation.
            </p>
          </div>
          <div className="text-xs tabular-nums text-[var(--faint)] shrink-0">
            PyTorch 2-Layer LSTM · 30,000 Vocab
          </div>
        </div>

        {/* Navigation Tabs with 1px accent indicator line */}
        <nav className="flex items-center gap-1 sm:gap-2 border-b border-[var(--border)] overflow-x-auto pb-px" aria-label="Workbench tabs">
          <button
            onClick={() => setActiveTab("predict")}
            className={`px-3 py-2 text-[13px] font-medium transition-colors duration-150 cursor-pointer shrink-0 ${
              activeTab === "predict"
                ? "text-[var(--text)] border-b-2 border-[var(--accent)] font-semibold"
                : "text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            Autocomplete
          </button>
          <button
            onClick={() => setActiveTab("generate")}
            className={`px-3 py-2 text-[13px] font-medium transition-colors duration-150 cursor-pointer shrink-0 ${
              activeTab === "generate"
                ? "text-[var(--text)] border-b-2 border-[var(--accent)] font-semibold"
                : "text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            Generation
          </button>
          <button
            onClick={() => setActiveTab("evaluation")}
            className={`px-3 py-2 text-[13px] font-medium transition-colors duration-150 cursor-pointer shrink-0 ${
              activeTab === "evaluation"
                ? "text-[var(--text)] border-b-2 border-[var(--accent)] font-semibold"
                : "text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            Multi-Domain Evaluation
          </button>
          <button
            onClick={() => setActiveTab("how-it-works")}
            className={`px-3 py-2 text-[13px] font-medium transition-colors duration-150 cursor-pointer shrink-0 ${
              activeTab === "how-it-works"
                ? "text-[var(--text)] border-b-2 border-[var(--accent)] font-semibold"
                : "text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            Architecture & Flow
          </button>
        </nav>

        {/* Active Tab View */}
        <div>
          {activeTab === "predict" && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              <div className="lg:col-span-8">
                <PredictionSection
                  initialText={selectedPrompt}
                  onSuccess={(prompt, result) => handleAddHistory("prediction", prompt, result)}
                />
              </div>
              <div className="lg:col-span-4 space-y-6">
                <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-5 space-y-3">
                  <div className="pb-2 border-b border-[var(--rule)]">
                    <h2 className="text-sm font-semibold text-[var(--text)]">
                      Active Model Telemetry
                    </h2>
                    <p className="text-xs text-[var(--muted)] mt-0.5">
                      WikiText-103 Causal Language Model
                    </p>
                  </div>
                  <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                    <div>
                      <dt className="text-[var(--muted)]">Architecture</dt>
                      <dd className="font-mono text-[var(--text)]">2-Layer LSTM</dd>
                    </div>
                    <div>
                      <dt className="text-[var(--muted)]">Vocabulary</dt>
                      <dd className="font-mono tabular-nums text-[var(--text)]">30,000 tokens</dd>
                    </div>
                    <div>
                      <dt className="text-[var(--muted)]">Embedding Dim</dt>
                      <dd className="font-mono tabular-nums text-[var(--text)]">256</dd>
                    </div>
                    <div>
                      <dt className="text-[var(--muted)]">Hidden Dim</dt>
                      <dd className="font-mono tabular-nums text-[var(--text)]">512</dd>
                    </div>
                    <div>
                      <dt className="text-[var(--muted)]">Context Window</dt>
                      <dd className="font-mono tabular-nums text-[var(--text)]">35 tokens</dd>
                    </div>
                    <div>
                      <dt className="text-[var(--muted)]">Dropout</dt>
                      <dd className="font-mono tabular-nums text-[var(--text)]">0.30</dd>
                    </div>
                  </dl>
                </div>

                <HistoryPanel
                  history={history}
                  onSelectPrompt={handleSelectHistoryPrompt}
                  onClearHistory={handleClearHistory}
                />
              </div>
            </div>
          )}

          {activeTab === "generate" && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              <div className="lg:col-span-8">
                <GenerationSection
                  initialPrompt={selectedPrompt}
                  onSuccess={(prompt, result) => handleAddHistory("generation", prompt, result)}
                />
              </div>
              <div className="lg:col-span-4 space-y-6">
                <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-5 space-y-3">
                  <div className="pb-2 border-b border-[var(--rule)]">
                    <h2 className="text-sm font-semibold text-[var(--text)]">
                      Sampling Controls Guide
                    </h2>
                    <p className="text-xs text-[var(--muted)] mt-0.5">
                      Autoregressive decoding parameters
                    </p>
                  </div>
                  <dl className="space-y-2 text-xs">
                    <div>
                      <dt className="font-medium text-[var(--text)]">Temperature (T)</dt>
                      <dd className="text-[var(--muted)]">Scales logits before softmax. T &lt; 1 sharpens distribution; T &gt; 1 flattens.</dd>
                    </div>
                    <div>
                      <dt className="font-medium text-[var(--text)]">Top-K Filtering</dt>
                      <dd className="text-[var(--muted)]">Restricts candidate sampling pool to the K highest probability tokens.</dd>
                    </div>
                    <div>
                      <dt className="font-medium text-[var(--text)]">Nucleus (Top-P)</dt>
                      <dd className="text-[var(--muted)]">Accumulates cumulative probability mass P before sampling.</dd>
                    </div>
                  </dl>
                </div>

                <HistoryPanel
                  history={history}
                  onSelectPrompt={handleSelectHistoryPrompt}
                  onClearHistory={handleClearHistory}
                />
              </div>
            </div>
          )}

          {activeTab === "evaluation" && (
            <div className="w-full">
              <GeneralizationSection />
            </div>
          )}

          {activeTab === "how-it-works" && (
            <div className="w-full">
              <HowItWorks />
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--border)] bg-[var(--surface)] mt-auto py-4">
        <div className="max-w-6xl w-full mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-[var(--faint)]">
          <div>LSTM Causal Language Modeling Workbench · WikiText-103 Baseline & Adaptation</div>
          <div className="tabular-nums font-mono text-xs">Context: 35 · Hidden: 512 · Embedding: 256</div>
        </div>
      </footer>
    </div>
  );
}
