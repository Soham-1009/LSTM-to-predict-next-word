"use client";

import React from "react";

export default function HowItWorks() {
  const steps = [
    {
      num: "01",
      title: "Input Text & Tokenization",
      tag: "Preprocessing",
      desc: "The prompt string is normalized and split into distinct word and punctuation tokens, which are mapped to numerical IDs using a fixed vocabulary.",
    },
    {
      num: "02",
      title: "Word Embedding (256d)",
      tag: "Vector Space",
      desc: "Discrete token indices are converted into dense 256-dimensional vector representations capturing semantic relationships.",
    },
    {
      num: "03",
      title: "2-Layer LSTM (512d Hidden)",
      tag: "Sequential Modeling",
      desc: "Stacked recurrent layers process the sequence sequentially with forget, input, and output gates. 30% dropout prevents co-adaptation.",
    },
    {
      num: "04",
      title: "Linear Projection & Logits",
      tag: "Classification",
      desc: "The final hidden state vector is projected through a linear layer back into the full vocabulary space (~30k–50k logits).",
    },
    {
      num: "05",
      title: "Softmax & Sampling Strategy",
      tag: "Inference Controls",
      desc: "Logits are transformed into probability distributions. Temperature scales peakiness; Top-K and Top-P (nucleus) eliminate the long tail.",
    },
    {
      num: "06",
      title: "Decoding & Autoregressive Loop",
      tag: "Generation",
      desc: "The chosen token ID is decoded back to text. In generation mode, the token is appended to the context for the next recursive step until EOS.",
    },
  ];

  return (
    <div className="bg-white/85 dark:bg-neutral-900/60 border border-slate-200/90 dark:border-neutral-800/80 rounded-2xl p-6 shadow-xl space-y-6 transition-colors duration-200">
      <div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
          <h2 className="text-lg font-semibold text-slate-900 dark:text-neutral-100">How It Works</h2>
        </div>
        <p className="text-xs text-slate-500 dark:text-neutral-400 mt-1">
          End-to-end execution flow of the LSTM Causal Language Model inference pipeline.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {steps.map((step) => (
          <div
            key={step.num}
            className="p-4 bg-slate-50 dark:bg-neutral-950/70 border border-slate-200 dark:border-neutral-800/70 rounded-xl space-y-2 hover:border-slate-300 dark:hover:border-neutral-700 transition-colors"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold text-cyan-700 dark:text-cyan-400 bg-cyan-50 dark:bg-cyan-950/50 px-2 py-0.5 rounded border border-cyan-200 dark:border-cyan-800/50">
                {step.num}
              </span>
              <span className="text-[10px] uppercase font-mono text-slate-600 dark:text-neutral-400 bg-slate-200 dark:bg-neutral-800/60 px-2 py-0.5 rounded">
                {step.tag}
              </span>
            </div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-neutral-200">{step.title}</h3>
            <p className="text-xs text-slate-600 dark:text-neutral-400 leading-relaxed">{step.desc}</p>
          </div>
        ))}
      </div>

      {/* Pipeline Diagram */}
      <div className="p-4 bg-slate-50 dark:bg-neutral-950/90 border border-slate-200 dark:border-neutral-800/70 rounded-xl overflow-x-auto">
        <div className="flex items-center justify-between min-w-[600px] text-xs font-mono text-slate-700 dark:text-neutral-300">
          <span className="px-2 py-1 bg-white dark:bg-neutral-800 rounded border border-slate-200 dark:border-neutral-700">Raw Text</span>
          <span className="text-slate-400 dark:text-neutral-500">→</span>
          <span className="px-2 py-1 bg-white dark:bg-neutral-800 rounded border border-slate-200 dark:border-neutral-700">Tokenizer</span>
          <span className="text-slate-400 dark:text-neutral-500">→</span>
          <span className="px-2 py-1 bg-white dark:bg-neutral-800 rounded border border-slate-200 dark:border-neutral-700">Embed (256)</span>
          <span className="text-slate-400 dark:text-neutral-500">→</span>
          <span className="px-2 py-1 bg-white dark:bg-neutral-800 rounded border border-slate-200 dark:border-neutral-700">2x LSTM (512)</span>
          <span className="text-slate-400 dark:text-neutral-500">→</span>
          <span className="px-2 py-1 bg-white dark:bg-neutral-800 rounded border border-slate-200 dark:border-neutral-700">Linear (Vocab)</span>
          <span className="text-slate-400 dark:text-neutral-500">→</span>
          <span className="px-2 py-1 bg-cyan-100 dark:bg-cyan-950 text-cyan-800 dark:text-cyan-300 rounded border border-cyan-200 dark:border-cyan-700">Sample / Softmax</span>
        </div>
      </div>
    </div>
  );
}
