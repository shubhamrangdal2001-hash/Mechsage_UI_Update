import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EngineCard } from "@/components/organisms/EngineCard";
import { cyclePayload } from "./fixtures";

describe("EngineCard", () => {
  it("renders the public Ironside identity and model output", () => {
    render(<EngineCard payload={cyclePayload} />);
    expect(screen.getByRole("link", { name: "Open diagnostics for ISM-CNC-001" })).toHaveAttribute("href", "/engines/FD001");
    expect(screen.getByText("ISM-CNC-001")).toBeInTheDocument();
    expect(screen.getByText("5-Axis CNC Machining Center")).toBeInTheDocument();
    expect(screen.queryByText(/SIM-FD|Turbofan/i)).not.toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("0.210")).toBeInTheDocument();
  });
});
