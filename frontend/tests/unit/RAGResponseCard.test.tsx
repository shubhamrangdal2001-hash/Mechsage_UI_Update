import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RAGResponseCard } from "@/components/organisms/RAGResponseCard";
import type { RAGQueryResponse } from "@/types";

const response: RAGQueryResponse = {
  query: "ISM-GBX-003 gear pitting",
  answer: "Inspect ISM-GBX-003 gear mesh and oil condition.",
  abstained: false,
  relevance_score: 0.92,
  threshold_used: 0.3,
  sources: [{ doc_id: "ISM-GBX-003-PROFILE", text: "Ironside gearbox profile", relevance_score: 0.92, source_file: "Ironside Manufacturing knowledge base" }],
  corpus_version: "2.0.0-ironside",
  retrieval_mode: "resource_safe_lexical",
};

describe("RAGResponseCard", () => {
  it("renders grounded answer and citations", () => {
    render(<RAGResponseCard response={response} />);
    expect(screen.getByText(/Inspect ISM-GBX-003/)).toBeInTheDocument();
    expect(screen.getByText("ISM-GBX-003-PROFILE")).toBeInTheDocument();
    expect(screen.queryByText(/turbofan/i)).not.toBeInTheDocument();
  });

  it("renders guardrail abstention without citations", () => {
    render(<RAGResponseCard response={{ ...response, answer: "", abstained: true, abstain_reason: "Off domain", relevance_score: 0, sources: [] }} />);
    expect(screen.getByText("Guardrail Triggered")).toBeInTheDocument();
    expect(screen.getByText(/Off domain/)).toBeInTheDocument();
    expect(screen.queryByText(/Citations/)).not.toBeInTheDocument();
  });
});
