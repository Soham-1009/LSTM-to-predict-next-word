import { Card } from "@/components/ui/Card";

const STEPS = [
  { title: "Input Text", desc: "The user types a sentence or fragment." },
  { title: "Tokenization", desc: "Text is normalized and split into word tokens." },
  { title: "Integer Encoding", desc: "Each token is mapped to a vocabulary id." },
  { title: "Embedding", desc: "Ids become dense vectors capturing word meaning." },
  { title: "LSTM", desc: "Sequential context is encoded into a hidden state." },
  { title: "Linear Layer", desc: "Hidden state is projected onto the vocabulary." },
  { title: "Softmax", desc: "Logits become a probability distribution." },
  { title: "Next Word", desc: "The argmax (or sampled) token is returned." },
] as const;

const WHY_LSTM = [
  "LSTMs are recurrent networks designed for sequences.",
  "They maintain a memory cell that carries context across time steps.",
  "Gates (input, forget, output) learn what to remember or forget.",
  "They reduce the vanishing-gradient problem of vanilla RNNs.",
  "They are well-suited to language modeling on modest corpora.",
] as const;

export function HowItWorksPanel() {
  return (
    <Card aria-labelledby="how-it-works-heading">
      <h2 id="how-it-works-heading" className="text-lg font-semibold text-ink">
        How the Model Works
      </h2>
      <p className="mt-1 text-sm text-ink-muted">
        A PyTorch LSTM is fed the last <span className="font-mono">N</span> tokens and outputs a
        probability distribution over the entire vocabulary.
      </p>

      <ol className="mt-5 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step, idx) => (
          <li
            key={step.title}
            className="relative rounded-xl border border-surface-border bg-surface-muted p-3"
          >
            <span className="absolute -top-2 left-3 rounded-full bg-brand-600 px-2 py-0.5 text-[10px] font-semibold text-white dark:bg-brand-500">
              {idx + 1}
            </span>
            <p className="mt-1 text-sm font-semibold text-ink">{step.title}</p>
            <p className="mt-1 text-xs text-ink-muted">{step.desc}</p>
          </li>
        ))}
      </ol>

      <div className="mt-6 rounded-xl border border-brand-100 bg-brand-50 p-4 dark:border-brand-800/60 dark:bg-brand-900/30">
        <h3 className="text-sm font-semibold text-brand-800 dark:text-brand-200">Why LSTM?</h3>
        <ul className="mt-2 space-y-1.5 text-sm text-brand-900 dark:text-brand-100">
          {WHY_LSTM.map((line) => (
            <li key={line} className="flex gap-2">
              <span
                className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-brand-500"
                aria-hidden="true"
              />
              <span>{line}</span>
            </li>
          ))}
        </ul>
      </div>
    </Card>
  );
}
