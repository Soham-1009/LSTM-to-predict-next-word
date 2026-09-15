import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api";
import { API_BASE_URL } from "../config";

const originalFetch = global.fetch;

function mockJsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("api client", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("hits the configured base URL for health", async () => {
    (global.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      mockJsonResponse({
        status: "healthy",
        model_loaded: true,
        tokenizer_loaded: true,
        device: "cpu",
      }),
    );

    const result = await api.healthCheck();
    expect(result.status).toBe("healthy");
    expect(global.fetch).toHaveBeenCalledWith(
      `${API_BASE_URL}/health`,
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("serializes predict request body to match the backend schema", async () => {
    (global.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      mockJsonResponse({
        input_text: "hello world",
        predictions: [
          { word: "there", probability: 0.6 },
          { word: "world", probability: 0.4 },
        ],
      }),
    );

    const result = await api.predictNextWord({ text: "hello world", topK: 2 });
    expect(result.predictions[0]?.word).toBe("there");

    const [url, init] = (global.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe(`${API_BASE_URL}/predict`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ text: "hello world", top_k: 2 });
  });

  it("serializes generate request body with the right field names", async () => {
    (global.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      mockJsonResponse({
        input_text: "once upon",
        generated_text: "once upon a time",
        generated_words: ["a", "time"],
      }),
    );

    await api.generateText({
      text: "once upon",
      numWords: 2,
      temperature: 0.7,
      strategy: "top_k",
    });

    const [, init] = (global.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      text: "once upon",
      num_words: 2,
      temperature: 0.7,
      strategy: "top_k",
    });
  });

  it("throws an ApiError with the backend detail on 4xx", async () => {
    (global.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response(JSON.stringify({ detail: "Input text cannot be empty." }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    );

    let captured: unknown = null;
    try {
      await api.predictNextWord({ text: "", topK: 5 });
    } catch (err) {
      captured = err;
    }
    expect(captured).toBeInstanceOf(ApiError);
    expect((captured as ApiError).message).toContain("empty");
    expect((captured as ApiError).status).toBe(400);
  });

  it("throws a network-flavoured ApiError when fetch rejects", async () => {
    (global.fetch as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(
      new TypeError("Failed to fetch"),
    );

    await expect(api.healthCheck()).rejects.toMatchObject({
      status: 0,
    });
  });
});
