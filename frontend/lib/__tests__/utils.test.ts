import { describe, expect, it } from "vitest";
import { classNames, formatPercent, formatTime, truncate } from "../utils";

describe("formatPercent", () => {
  it("formats probabilities with one decimal", () => {
    expect(formatPercent(0.724)).toBe("72.4%");
    expect(formatPercent(0.5)).toBe("50.0%");
  });

  it("returns em-dash for invalid input", () => {
    expect(formatPercent(null)).toBe("—");
    expect(formatPercent(undefined)).toBe("—");
    expect(formatPercent(Number.NaN)).toBe("—");
  });
});

describe("formatTime", () => {
  it("returns empty for falsy values", () => {
    expect(formatTime(null)).toBe("");
    expect(formatTime(undefined)).toBe("");
    expect(formatTime(0)).toBe("");
  });

  it("returns a non-empty string for a real timestamp", () => {
    expect(formatTime(1700000000000)).not.toBe("");
  });
});

describe("truncate", () => {
  it("returns the original string when under the limit", () => {
    expect(truncate("hi", 10)).toBe("hi");
  });

  it("truncates with ellipsis when over the limit", () => {
    const result = truncate("a".repeat(50), 10);
    expect(result.length).toBe(10);
    expect(result.endsWith("…")).toBe(true);
  });

  it("handles empty input", () => {
    expect(truncate("")).toBe("");
  });
});

describe("classNames", () => {
  it("joins truthy class names", () => {
    expect(classNames("a", "b", "c")).toBe("a b c");
  });

  it("filters out falsy values", () => {
    expect(classNames("a", false, null, undefined, "b")).toBe("a b");
  });
});
