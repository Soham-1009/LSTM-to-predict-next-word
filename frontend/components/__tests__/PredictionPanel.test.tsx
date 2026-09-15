import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { PredictionPanel } from "@/components/prediction/PredictionPanel";

function defaultProps() {
  return {
    text: "hello",
    onTextChange: vi.fn(),
    onClear: vi.fn(),
    onSubmit: vi.fn(),
    loading: false,
    disabled: false,
    topK: 5,
    onTopKChange: vi.fn(),
  };
}

describe("PredictionPanel", () => {
  it("calls onSubmit when the predict button is clicked", () => {
    const props = defaultProps();
    render(<PredictionPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: /predict next word/i }));
    expect(props.onSubmit).toHaveBeenCalledTimes(1);
  });

  it("disables the predict button when input is empty", () => {
    const props = { ...defaultProps(), text: "" };
    render(<PredictionPanel {...props} />);
    const button = screen.getByRole("button", { name: /predict next word/i });
    expect(button).toBeDisabled();
  });

  it("disables the predict button when disabled=true", () => {
    const props = { ...defaultProps(), disabled: true };
    render(<PredictionPanel {...props} />);
    const button = screen.getByRole("button", { name: /predict next word/i });
    expect(button).toBeDisabled();
  });

  it("updates the textarea value on change", () => {
    const props = defaultProps();
    render(<PredictionPanel {...props} />);
    const textarea = screen.getByLabelText(/enter a sentence/i) as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "new text" } });
    expect(props.onTextChange).toHaveBeenCalledWith("new text");
  });

  it("calls onTopKChange when a top-k button is clicked", () => {
    const props = defaultProps();
    render(<PredictionPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "3" }));
    expect(props.onTopKChange).toHaveBeenCalledWith(3);
  });

  it("populates the input when an example is clicked", () => {
    const props = defaultProps();
    render(<PredictionPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: /Artificial intelligence is/ }));
    expect(props.onTextChange).toHaveBeenCalledWith("Artificial intelligence is");
  });
});
