// frontend/tests/unit/MetricCard.test.tsx
import React from "react";
import { render, screen } from "@testing-library/react";
import { MetricCard } from "../../components/molecules/MetricCard";

describe("MetricCard Component", () => {
  it("renders metric title and value correctly", () => {
    render(
      <MetricCard
        title="Active Engines"
        value="4"
        description="Systems online"
      />
    );

    expect(screen.getByText("Active Engines")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("Systems online")).toBeInTheDocument();
  });

  it("applies correct glowing class when glow prop is set", () => {
    const { container } = render(
      <MetricCard
        title="Critical Alerts"
        value="2"
        description="Shutdown required"
        glow="red"
      />
    );

    const card = container.firstChild;
    expect(card).toHaveClass("card-glow-red");
  });
});
