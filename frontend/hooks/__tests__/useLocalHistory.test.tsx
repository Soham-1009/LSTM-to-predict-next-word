import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useLocalHistory } from "@/hooks/useLocalHistory";
import { HISTORY_STORAGE_KEY } from "@/lib/config";

describe("useLocalHistory", () => {
  beforeEach(() => {
    if (typeof window !== "undefined" && window.localStorage) {
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
    }
  });

  it("starts with an empty history", () => {
    const { result } = renderHook(() => useLocalHistory());
    expect(result.current.history).toEqual([]);
  });

  it("adds entries and persists to localStorage", () => {
    const { result } = renderHook(() => useLocalHistory());

    act(() => {
      result.current.addEntry({
        kind: "predict",
        input: "hello",
        output: "world",
      });
    });

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0]?.input).toBe("hello");
    const stored = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    expect(stored).toContain("hello");
  });

  it("removes an entry by id", () => {
    const { result } = renderHook(() => useLocalHistory());
    act(() => {
      result.current.addEntry({ kind: "predict", input: "a", output: "b" });
      result.current.addEntry({ kind: "predict", input: "c", output: "d" });
    });
    // After two adds, the newest entry ("c") is at index 0.
    expect(result.current.history[0]?.input).toBe("c");
    expect(result.current.history[1]?.input).toBe("a");
    const idToRemove = result.current.history[0]?.id;
    act(() => {
      if (idToRemove) result.current.removeEntry(idToRemove);
    });
    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0]?.input).toBe("a");
  });

  it("clears all entries", () => {
    const { result } = renderHook(() => useLocalHistory());
    act(() => {
      result.current.addEntry({ kind: "predict", input: "a", output: "b" });
      result.current.clear();
    });
    expect(result.current.history).toEqual([]);
  });
});
