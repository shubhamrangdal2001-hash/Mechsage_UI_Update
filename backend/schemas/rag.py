"""
backend/schemas/rag.py
----------------------
Pydantic v2 schemas for the RAG pipeline request/response.
Includes abstain state and source citations for frontend display.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RAGSource(BaseModel):
    """A single retrieved and re-ranked document source."""
    doc_id: str
    text: str
    relevance_score: float
    source_file: Optional[str] = None
    page: Optional[int] = None


class RAGQueryRequest(BaseModel):
    """Query body for the RAG pipeline."""
    query: str = Field(..., min_length=3, description="Maintenance query in plain English")
    top_k: int = Field(5, ge=1, le=20)
    dataset_context: Optional[str] = Field(
        None, description="Optional: FD001–FD004 to restrict retrieval context"
    )


class RAGQueryResponse(BaseModel):
    """Full RAG pipeline response with answer, sources, and guardrail state."""
    query: str
    answer: str
    abstained: bool = Field(
        ..., description="True if the reranker score fell below the threshold"
    )
    abstain_reason: Optional[str] = Field(
        None, description="Reason for abstaining, if abstained=True"
    )
    relevance_score: float = Field(..., description="Top reranker score [0–1]")
    threshold_used: float
    sources: List[RAGSource]
    latency_ms: Optional[float] = None
    corpus_version: str
    retrieval_mode: str
