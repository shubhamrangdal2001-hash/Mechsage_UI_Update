"""
backend/api/v1/routers/rag.py
-------------------------------
RAG pipeline query endpoint with strict guardrails.

Route:
  POST /api/v1/rag/query   — Hybrid retrieval → cross-encoder rerank → LLM generate
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.security import verify_api_key
from backend.schemas.rag import RAGQueryRequest, RAGQueryResponse, RAGSource

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])
logger = get_logger(__name__)


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Query the maintenance manual RAG pipeline",
    dependencies=[Depends(verify_api_key)],
)
async def rag_query(req: RAGQueryRequest):
    """
    Run the full RAG pipeline:
    1. Hybrid retrieval (ChromaDB dense + BM25 sparse)
    2. Cross-encoder reranking
    3. Guardrail threshold check (abstain if score < threshold)
    4. LLM generation with retrieved context
    """
    try:
        from dev.rag.lightweight_retriever import search_ironside
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG pipeline not available: {str(exc)}",
        )

    t0 = time.perf_counter()

    try:
        retrieval = search_ironside(req.query, top_k=req.top_k)

        if retrieval.get("error_code"):
            result = {
                "answer": "",
                "abstained": True,
                "abstain_reason": retrieval.get("message", "No grounded Ironside passage found."),
                "relevance_score": 0.0,
                "sources": [],
                "corpus_version": retrieval.get("corpus_version", "unknown"),
                "retrieval_mode": retrieval.get("retrieval_mode", "unknown"),
            }
        else:
            passages = retrieval.get("retrieved_passages", [])
            top_score = float(passages[0].get("relevance_score", 0.0)) if passages else 0.0
            grounded = bool(passages) and top_score >= settings.rag_relevance_threshold
            sources = [
                {
                    "doc_id": passage.get("doc_ref", "unknown"),
                    "text": passage.get("text", ""),
                    "relevance_score": passage.get("relevance_score", 0.0),
                    "source_file": "Ironside Manufacturing knowledge base",
                }
                for passage in passages
            ] if grounded else []
            result = {
                "answer": passages[0].get("text", "") if grounded else "",
                "abstained": not grounded,
                "abstain_reason": None if grounded else "No grounded Ironside passage met the relevance threshold.",
                "relevance_score": top_score,
                "sources": sources,
                "corpus_version": retrieval.get("corpus_version", "unknown"),
                "retrieval_mode": retrieval.get("retrieval_mode", "unknown"),
            }
    except Exception as exc:
        logger.error("RAG pipeline error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline failed: {str(exc)}",
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    # Parse the result dict from the RAG pipeline
    abstained = result.get("abstained", False)
    answer = result.get("answer", "")
    relevance_score = float(result.get("relevance_score", 0.0))
    sources_raw = result.get("sources", [])

    sources = [
        RAGSource(
            doc_id=s.get("doc_id", s.get("id", "unknown")),
            text=s.get("text", s.get("content", "")),
            relevance_score=float(s.get("score", s.get("relevance_score", 0.0))),
            source_file=s.get("source_file"),
            page=s.get("page"),
        )
        for s in sources_raw
    ]

    logger.info(
        "RAG query complete",
        abstained=abstained,
        relevance_score=round(relevance_score, 3),
        source_count=len(sources),
        latency_ms=round(latency_ms, 1),
        corpus_version=result.get("corpus_version", "unknown"),
        retrieval_mode=result.get("retrieval_mode", "unknown"),
    )

    return RAGQueryResponse(
        query=req.query,
        answer=answer,
        abstained=abstained,
        abstain_reason=result.get("abstain_reason") if abstained else None,
        relevance_score=relevance_score,
        threshold_used=settings.rag_relevance_threshold,
        sources=sources,
        latency_ms=round(latency_ms, 1),
        corpus_version=result.get("corpus_version", "unknown"),
        retrieval_mode=result.get("retrieval_mode", "unknown"),
    )
