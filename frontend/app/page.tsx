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
    setHistory((prev) => [newItem, ...prev.slice(0, 19)]); // Keep last 20 items
  };

  const handleSelectHistoryPrompt = (prompt: string, type: "prediction" | "generation") => {
    setSelectedPrompt(prompt);
    setActiveTab(type === "prediction" ? "predict" : "generate");
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-slate-100 dark:bg-neutral-950 text-slate-900 dark:text-neutral-100 flex flex-col selection:bg-cyan-500/30 selection:text-cyan-200 transition-colors duration-200">
      {/* Top Header */}
      <header className="border-b border-slate-200/90 dark:border-neutral-800/80 bg-white/80 dark:bg-neutral-950/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-500 to-violet-600 flex items-center justify-center font-bold text-white text-sm shadow-md shadow-cyan-500/20">
              GA
            </div>
            <div>
              <h1 className="font-semibold text-sm tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
                LSTM Language Model
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-200 dark:bg-neutral-800 text-slate-700 dark:text-neutral-400">
                  v0.1
                </span>
              </h1>
              <p className="text-[11px] text-slate-500 dark:text-neutral-400 hidden sm:block">
                Causal next-word prediction & autoregressive text generator
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2.5">
            <ModelInfoBadge />
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 flex flex-col justify-center max-w-5xl w-full mx-auto px-4 sm:px-6 py-6">
        <div className="w-full space-y-6 my-auto">
          {/* Navigation Tabs */}
          <div className="flex items-center justify-center">
            <div className="inline-flex p-1 bg-white/90 dark:bg-neutral-900/90 border border-slate-200/90 dark:border-neutral-800 rounded-xl shadow-inner gap-1">
              <button
                onClick={() => setActiveTab("predict")}
                className={`px-4 sm:px-5 py-2 rounded-lg text-xs font-medium transition-all ${
                  activeTab === "predict"
                    ? "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30 shadow-sm"
                    : "text-slate-600 dark:text-neutral-400 hover:text-slate-900 dark:hover:text-neutral-200"
                }`}
              >
                Interactive Autocomplete
              </button>
              <button
                onClick={() => setActiveTab("generate")}
                className={`px-4 sm:px-5 py-2 rounded-lg text-xs font-medium transition-all ${
                  activeTab === "generate"
                    ? "bg-violet-500/15 text-violet-700 dark:text-violet-300 border border-violet-500/30 shadow-sm"
                    : "text-slate-600 dark:text-neutral-400 hover:text-slate-900 dark:hover:text-neutral-200"
                }`}
              >
                Autoregressive Generation
              </button>
              <button
                onClick={() => setActiveTab("evaluation")}
                className={`px-4 sm:px-5 py-2 rounded-lg text-xs font-medium transition-all ${
                  activeTab === "evaluation"
                    ? "bg-cyan-600/20 text-cyan-700 dark:text-cyan-300 border border-cyan-500/40 shadow-sm font-semibold"
                    : "text-slate-600 dark:text-neutral-400 hover:text-slate-900 dark:hover:text-neutral-200"
                }`}
              >
                Multi-Domain Evaluation
              </button>
              <button
                onClick={() => setActiveTab("how-it-works")}
                className={`px-4 sm:px-5 py-2 rounded-lg text-xs font-medium transition-all ${
                  activeTab === "how-it-works"
                    ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/30 shadow-sm"
                    : "text-slate-600 dark:text-neutral-400 hover:text-slate-900 dark:hover:text-neutral-200"
                }`}
              >
                How It Works
              </button>
            </div>
          </div>

          {/* Dynamic Section View */}
          <div className="space-y-6">
            {activeTab === "predict" && (
              <PredictionSection
                initialText={selectedPrompt}
                onSuccess={(prompt, result) => handleAddHistory("prediction", prompt, result)}
              />
            )}

            {activeTab === "generate" && (
              <GenerationSection
                initialPrompt={selectedPrompt}
                onSuccess={(prompt, result) => handleAddHistory("generation", prompt, result)}
              />
            )}

            {activeTab === "evaluation" && <GeneralizationSection />}

            {activeTab === "how-it-works" && <HowItWorks />}

            {/* History Panel */}
            {activeTab !== "how-it-works" && activeTab !== "evaluation" && (
              <HistoryPanel
                history={history}
                onSelectPrompt={handleSelectHistoryPrompt}
                onClearHistory={handleClearHistory}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
