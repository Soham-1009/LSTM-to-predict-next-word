import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PredictionResult } from "@/components/prediction/PredictionResult";
import type { PredictResponse } from "@/types/api";

const sample: PredictResponse = {
  input_text: "hello world",
  predictions: [
    { word: "there", probability: 0.7 },
    { word: "again", probability: 0.3 },
  ],
};

describe("PredictionResult", () => {
  it("renders the top-1 word and confidence", () => {
    render(<PredictionResult data={sample} loading={false} error={null} />);
    // "there" appears in both the headline and the top-k row.
    expect(screen.getAllByText("there").length).toBeGreaterThan(0);
    // "70.0%" appears in both the confidence line and the bar row.
    expect(screen.getAllByText("70.0%").length).toBeGreaterThan(0);
  });

  it("renders all top-k predictions as bars", () => {
    render(<PredictionResult data={sample} loading={false} error={null} />);
    expect(screen.getByText("again")).toBeInTheDocument();
    expect(screen.getByText("30.0%")).toBeInTheDocument();
  });

  it("renders an empty-state message when no data is present", () => {
    render(<PredictionResult data={null} loading={false} error={null} />);
    expect(screen.getByText(/Predict Next Word/i)).toBeInTheDocument();
  });

  it("renders a loading indicator", () => {
    render(<PredictionResult data={null} loading={true} error={null} />);
    expect(screen.getByText(/Querying the LSTM/i)).toBeInTheDocument();
  });

  it("renders an error message", () => {
    render(<PredictionResult data={null} loading={false} error="Backend offline" />);
    expect(screen.getByText("Backend offline")).toBeInTheDocument();
  });
});
