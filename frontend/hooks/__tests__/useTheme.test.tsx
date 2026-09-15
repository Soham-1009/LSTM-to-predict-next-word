import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useTheme } from "@/hooks/useTheme";

describe("useTheme", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.classList.remove("dark");
  });

  afterEach(() => {
    document.documentElement.classList.remove("dark");
  });

  it("starts on light and exposes a toggle", () => {
    const { result } = renderHook(() => useTheme());
    // Hydrate from useEffect.
    act(() => {
      // no-op
    });
    expect(result.current.theme).toBe("light");
    expect(typeof result.current.toggle).toBe("function");
    expect(result.current.ready).toBe(true);
  });

  it("toggles the dark class on <html> and persists to localStorage", () => {
    const { result } = renderHook(() => useTheme());
    act(() => {
      // hydrate
    });
    act(() => {
      result.current.toggle();
    });
    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(window.localStorage.getItem("lstm-theme")).toBe("dark");

    act(() => {
      result.current.toggle();
    });
    expect(result.current.theme).toBe("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(window.localStorage.getItem("lstm-theme")).toBe("light");
  });

  it("setTheme applies a specific value", () => {
    const { result } = renderHook(() => useTheme());
    act(() => {
      // hydrate
    });
    act(() => {
      result.current.setTheme("dark");
    });
    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });
});
