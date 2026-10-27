"use client";

import React from "react";

export default function HowItWorks() {
  const stages = [
    {
      title: "Input Text & Boundary Tokenization",
      role: "Preprocessing",
      desc: "Raw text is normalized and split into distinct word and punctuation tokens. A fixed 30,000-token vocabulary maps strings into sequence indices.",
    },
    {
      title: "Dense Embedding Projection",
      role: "256 Dimensions",
      desc: "Discrete token indices are converted into dense continuous vector representations capturing semantic and syntactic distributional geometry.",
    },
    {
      title: "Stacked Recurrent Modeling",
      role: "2-Layer LSTM (512d Hidden)",
      desc: "Stacked recurrent layers with input, forget, and output gates propagate causal temporal dependencies with 0.3 dropout regularization.",
    },
    {
      title: "Linear Vocabulary Projection",
      role: "Logits Tensor",
      desc: "The final hidden state vector projects through a linear layer into raw, unnormalized logits across all 30,000 vocabulary tokens.",
    },
    {
      title: "Softmax Distribution & Sampling",
      role: "Probability Estimation",
      desc: "Logits are transformed into categorical probability distributions. Nucleus (Top-P), Top-K, and temperature scaling filter low-probability tails.",
    },
    {
      title: "Autoregressive Decoding Loop",
      role: "Sequence Emission",
      desc: "Sampled tokens are appended back to the context window (length 35) to predict successive words iteratively until an EOS token or length limit.",
    },
  ];

  const specs = [
    { param: "Model Architecture", value: "2-Layer Causal LSTM" },
    { param: "Embedding Dimension", value: "256" },
    { param: "Hidden Dimension", value: "512 units per layer" },
    { param: "Context Window", value: "35 tokens" },
    { param: "Vocabulary Size", value: "30,000 word-level tokens" },
    { param: "Base Training Corpus", value: "WikiText-103 (103M tokens)" },
    { param: "Dropout Rate", value: "0.3 (recurrent & embedding)" },
    { param: "Serving Framework", value: "FastAPI + PyTorch inference_mode" },
  ];

  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-5 space-y-6">
      <div className="pb-3 border-b border-[var(--rule)]">
        <h2 className="text-base font-semibold text-[var(--text)]">
          Language Model Architecture & Pipeline
        </h2>
        <p className="text-xs text-[var(--muted)] mt-0.5">
          End-to-end execution flow of the recurrent causal neural network from raw text input to next-token emission.
        </p>
      </div>

      {/* Stage Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {stages.map((stage) => (
          <div
            key={stage.title}
            className="p-3.5 bg-[var(--ground)] border border-[var(--border)] rounded space-y-1.5"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-[11px] font-medium text-[var(--accent)]">
                {stage.role}
              </span>
            </div>
            <h3 className="text-xs font-semibold text-[var(--text)]">
              {stage.title}
            </h3>
            <p className="text-xs text-[var(--muted)] leading-relaxed">
              {stage.desc}
            </p>
          </div>
        ))}
      </div>

      {/* Pipeline Diagram */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-[var(--text)]">
          Sequential Data Flow
        </h3>
        <div className="p-3.5 bg-[var(--ground)] border border-[var(--border)] rounded overflow-x-auto">
          <div className="flex items-center justify-between min-w-[620px] text-xs text-[var(--text)]">
            <span className="px-2 py-1 bg-[var(--surface)] border border-[var(--border)] rounded font-mono text-[11px]">
              Raw Text Context
            </span>
            <span className="text-[var(--faint)]">→</span>
            <span className="px-2 py-1 bg-[var(--surface)] border border-[var(--border)] rounded font-mono text-[11px]">
              Tokenizer (Vocab 30k)
            </span>
            <span className="text-[var(--faint)]">→</span>
            <span className="px-2 py-1 bg-[var(--surface)] border border-[var(--border)] rounded font-mono text-[11px]">
              Embedding (256d)
            </span>
            <span className="text-[var(--faint)]">→</span>
            <span className="px-2 py-1 bg-[var(--surface)] border border-[var(--border)] rounded font-mono text-[11px]">
              2× LSTM (512d)
            </span>
            <span className="text-[var(--faint)]">→</span>
            <span className="px-2 py-1 bg-[var(--surface)] border border-[var(--border)] rounded font-mono text-[11px]">
              Linear Logits
            </span>
            <span className="text-[var(--faint)]">→</span>
            <span className="px-2 py-1 bg-[var(--surface)] border border-[var(--accent)] text-[var(--accent)] rounded font-mono text-[11px] font-medium">
              Softmax / Sample
            </span>
          </div>
        </div>
      </div>

      {/* Model Specifications Table */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-[var(--text)]">
          System Specifications
        </h3>
        <div className="border border-[var(--border)] rounded overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-[var(--ground)] border-b border-[var(--border)]">
                <th className="py-2 px-3 font-medium text-[var(--muted)] text-left">
                  Specification Parameter
                </th>
                <th className="py-2 px-3 font-medium text-[var(--muted)] text-right">
                  Configuration Value
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--rule)]">
              {specs.map((item) => (
                <tr key={item.param} className="hover:bg-[var(--ground)]">
                  <td className="py-2 px-3 text-[var(--text)]">
                    {item.param}
                  </td>
                  <td className="py-2 px-3 text-right font-mono text-[12px] text-[var(--muted)]">
                    {item.value}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
