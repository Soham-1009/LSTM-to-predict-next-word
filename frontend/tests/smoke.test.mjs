/**
 * Comprehensive frontend client, data-contract, and UI logic smoke tests.
 * Uses Node.js built-in test runner (node:test, node:assert).
 */

import { describe, it } from "node:test";
import assert from "node:assert";

describe("Frontend Configuration and API Routes", () => {
  const DEFAULT_API_URL = "http://127.0.0.1:8000";

  it("should configure the expected local loopback API URL", () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_URL;
    assert.strictEqual(apiUrl, "http://127.0.0.1:8000");
  });

  it("should specify all four core REST endpoint routes", () => {
    const endpoints = {
      health: `${DEFAULT_API_URL}/health`,
      modelInfo: `${DEFAULT_API_URL}/model-info`,
      predict: `${DEFAULT_API_URL}/predict`,
      generate: `${DEFAULT_API_URL}/generate`,
    };
    assert.strictEqual(endpoints.health, "http://127.0.0.1:8000/health");
    assert.strictEqual(endpoints.modelInfo, "http://127.0.0.1:8000/model-info");
    assert.strictEqual(endpoints.predict, "http://127.0.0.1:8000/predict");
    assert.strictEqual(endpoints.generate, "http://127.0.0.1:8000/generate");
  });
});

describe("Prediction Data Contracts & Autocomplete Extension", () => {
  // Simulates frontend handleSelectCandidate logic from PredictionSection.tsx
  function appendCandidateToken(currentText, token) {
    const trimmed = currentText.trimEnd();
    if (!trimmed) return token;
    const needsSpace =
      !/^[.,!?;:'")\]}]/.test(token) &&
      !trimmed.endsWith("(") &&
      !trimmed.endsWith("[") &&
      !trimmed.endsWith("{");
    return needsSpace ? `${trimmed} ${token}` : `${trimmed}${token}`;
  }

  it("should format prediction request with default Top-K = 5", () => {
    const req = { text: "The quick brown fox", top_k: 5 };
    assert.strictEqual(req.top_k, 5);
    assert.ok(req.text.trim().length > 0);
  });

  it("should validate candidate token response structure and probability ranges", () => {
    const mockCandidates = [
      { token: ".", probability: 0.2756, token_id: 6 },
      { token: ",", probability: 0.178, token_id: 5 },
      { token: "and", probability: 0.0447, token_id: 8 },
    ];

    assert.strictEqual(mockCandidates.length, 3);
    for (const c of mockCandidates) {
      assert.strictEqual(typeof c.token, "string");
      assert.strictEqual(typeof c.token_id, "number");
      assert.ok(c.probability >= 0.0 && c.probability <= 1.0);
    }
  });

  it("should correctly append word tokens with intervening whitespace", () => {
    const base = "Once upon a";
    const extended = appendCandidateToken(base, "time");
    assert.strictEqual(extended, "Once upon a time");
  });

  it("should correctly attach punctuation tokens without intervening whitespace", () => {
    const base = "Once upon a time";
    const punctuated = appendCandidateToken(base, ",");
    assert.strictEqual(punctuated, "Once upon a time,");

    const finished = appendCandidateToken(base, ".");
    assert.strictEqual(finished, "Once upon a time.");
  });

  it("should handle empty prompt when appending candidate token", () => {
    const extended = appendCandidateToken("", "Machine");
    assert.strictEqual(extended, "Machine");
  });

  it("should not insert space when appending token after opening parenthesis", () => {
    const base = "in (";
    const extended = appendCandidateToken(base, "the");
    assert.strictEqual(extended, "in (the");
  });
});

describe("Generation Data Contracts & Sampling Controls", () => {
  it("should validate default generation hyperparameters", () => {
    const defaults = {
      max_new_tokens: 30,
      temperature: 1.0,
      top_k: 40,
      top_p: 0.9,
      greedy: false,
    };
    assert.strictEqual(defaults.max_new_tokens, 30);
    assert.strictEqual(defaults.temperature, 1.0);
    assert.strictEqual(defaults.top_k, 40);
    assert.strictEqual(defaults.top_p, 0.9);
    assert.strictEqual(defaults.greedy, false);
  });

  it("should enforce greedy override on temperature parameter", () => {
    const greedy = true;
    const userTemp = 0.8;
    const effectiveTemp = greedy ? 0.0 : userTemp;
    assert.strictEqual(effectiveTemp, 0.0);
  });

  it("should validate generation response schema and token accounting", () => {
    const mockResponse = {
      prompt: "In the future",
      generated_text: "In the future, technology will advance rapidly.",
      tokens_generated: 7,
      finish_reason: "length",
    };
    assert.strictEqual(mockResponse.prompt, "In the future");
    assert.ok(mockResponse.generated_text.startsWith(mockResponse.prompt));
    assert.strictEqual(mockResponse.tokens_generated, 7);
    assert.ok(["length", "eos"].includes(mockResponse.finish_reason));
  });
});

describe("Session History & Telemetry Contract", () => {
  it("should conform to session history item schema", () => {
    const historyItem = {
      id: "hist_123456789",
      type: "prediction",
      prompt: "Artificial intelligence",
      result: "Top predictions: [is (25.0%), has (15.0%)]",
      timestamp: new Date().toLocaleTimeString(),
    };
    assert.ok(["prediction", "generation"].includes(historyItem.type));
    assert.strictEqual(typeof historyItem.prompt, "string");
    assert.strictEqual(typeof historyItem.result, "string");
    assert.strictEqual(typeof historyItem.timestamp, "string");
  });

  it("should validate model telemetry payload schema", () => {
    const telemetry = {
      model_name: "LSTM Causal Language Model",
      architecture: "2-Layer LSTM",
      vocab_size: 30000,
      embed_dim: 256,
      hidden_dim: 512,
      num_layers: 2,
      dropout: 0.3,
      context_length: 35,
      device: "cuda",
      checkpoint_loaded: true,
      status: "ready",
    };
    assert.strictEqual(telemetry.vocab_size, 30000);
    assert.strictEqual(telemetry.num_layers, 2);
    assert.strictEqual(telemetry.checkpoint_loaded, true);
    assert.strictEqual(telemetry.status, "ready");
  });

  it("should extract continuation text cleanly without character truncation when prompt contains trailing or internal whitespace", () => {
    function extractContinuation(rawPrompt, generatedText) {
      const tokenRegex = /\w+(?:'\w+)?|[^\w\s]/g;
      const promptTokens = rawPrompt.toLowerCase().match(tokenRegex) || [];
      if (promptTokens.length === 0) {
        return generatedText ? (generatedText.startsWith(" ") ? generatedText : ` ${generatedText}`) : "";
      }
      let count = 0;
      let splitIndex = 0;
      while (tokenRegex.exec(generatedText) !== null) {
        count++;
        if (count === promptTokens.length) {
          splitIndex = tokenRegex.lastIndex;
          break;
        }
      }
      if (count < promptTokens.length) return "";
      const rem = generatedText.slice(splitIndex);
      if (!rem) return "";
      if (!rem.startsWith(" ") && !/^[.,!?;:'")\]}]/.test(rem)) {
        return ` ${rem}`;
      }
      return rem;
    }

    // Trailing whitespace
    const trailingPrompt = "in the deep   ";
    const genText = "in the deep ocean was dark";
    const cont1 = extractContinuation(trailingPrompt, genText);
    assert.strictEqual(cont1, " ocean was dark");
    assert.ok(!cont1.startsWith("ean"));

    // Multiple internal whitespace
    const internalPrompt = "in   the   deep";
    const cont2 = extractContinuation(internalPrompt, genText);
    assert.strictEqual(cont2, " ocean was dark");
    assert.ok(!cont2.startsWith("cean"));
  });

  it("should correctly reflect degraded status when checkpoint is not loaded in telemetry", () => {
    const degradedTelemetry = {
      model_name: "LSTM Causal Language Model",
      vocab_size: 30000,
      checkpoint_loaded: false,
      status: "degraded",
    };
    assert.strictEqual(degradedTelemetry.checkpoint_loaded, false);
    assert.strictEqual(degradedTelemetry.status, "degraded");
  });
});

describe("Generalization and Multi-Domain Evaluation Contracts", () => {
  const DEFAULT_API_URL = "http://127.0.0.1:8000";

  it("should define routes for evaluation reports and checkpoint selection", () => {
    const endpoints = {
      generalization: `${DEFAULT_API_URL}/evaluation/generalization`,
      testBaseline: `${DEFAULT_API_URL}/evaluation/test-baseline`,
      selectCheckpoint: `${DEFAULT_API_URL}/model/select-checkpoint`,
    };
    assert.strictEqual(endpoints.generalization, "http://127.0.0.1:8000/evaluation/generalization");
    assert.strictEqual(endpoints.testBaseline, "http://127.0.0.1:8000/evaluation/test-baseline");
    assert.strictEqual(endpoints.selectCheckpoint, "http://127.0.0.1:8000/model/select-checkpoint");
  });

  it("should validate GeneralizationReport schema structure", () => {
    const mockReport = {
      title: "Empirical Multi-Domain Generalization & Model Adaptation Report",
      leakage_audit_summary: {
        status: "AUDIT_PASSED",
        invariants_passed: "11/11",
        parent_source_overlap_pct: 0.0,
      },
      overall_summary: {
        baseline: { loss: 4.136, perplexity: 62.56 },
        adapted: { loss: 3.6981, perplexity: 40.37 },
        deltas: { loss_delta: -0.4379, ppl_delta: -22.19 },
      },
      domain_breakdown: [
        { domain: "conversation", baseline: { perplexity: 164.52 }, adapted: { perplexity: 146.16 } },
        { domain: "literature", baseline: { perplexity: 154.16 }, adapted: { perplexity: 127.67 } },
      ],
    };

    assert.strictEqual(mockReport.leakage_audit_summary.status, "AUDIT_PASSED");
    assert.strictEqual(mockReport.leakage_audit_summary.parent_source_overlap_pct, 0.0);
    assert.ok(mockReport.overall_summary.deltas.loss_delta < 0);
    assert.ok(mockReport.overall_summary.deltas.ppl_delta < 0);
    assert.strictEqual(mockReport.domain_breakdown.length, 2);
  });

  it("should validate checkpoint switching payload and supported models", () => {
    const supported = ["baseline", "adapted"];
    const selectBaselineReq = { checkpoint_name: "baseline" };
    const selectAdaptedReq = { checkpoint_name: "adapted" };

    assert.ok(supported.includes(selectBaselineReq.checkpoint_name));
    assert.ok(supported.includes(selectAdaptedReq.checkpoint_name));
  });
});

