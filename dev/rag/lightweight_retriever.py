"""Resource-safe lexical retrieval over the canonical MechSage knowledge base."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

_KB_PATH = Path(__file__).parent / "knowledge_base" / "knowledge_base_v2.json"
_IRONSIDE_PREFIXES = ("ISM-", "SOP-", "ANOMALY-", "WO-", "RUL-", "HIST-", "SPARE-")
_STOP_WORDS = {
    "a", "an", "and", "are", "for", "how", "in", "is", "of", "on", "the",
    "to", "what", "when", "with", "should", "required",
}


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOP_WORDS
    }


@lru_cache(maxsize=1)
def _entries() -> list[dict]:
    with _KB_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def search_ironside(query: str, top_k: int = 3) -> dict:
    """Return grounded Ironside passages without loading transformer models."""
    query_tokens = _tokens(query)
    query_lower = query.lower()
    query_asset_ids = set(re.findall(r"ism-[a-z]+-\d+", query_lower))
    ranked: list[tuple[float, dict]] = []

    for entry in _entries():
        entry_id = str(entry.get("id", ""))
        if entry.get("asset_type") != "ironside" and not entry_id.startswith(_IRONSIDE_PREFIXES):
            continue

        searchable = " ".join([
            entry_id,
            str(entry.get("fault_mode", "")),
            " ".join(entry.get("components", [])),
            " ".join(entry.get("sensor_cues", [])),
            str(entry.get("text", "")),
        ])
        document_tokens = _tokens(searchable)
        overlap = len(query_tokens & document_tokens)
        if overlap == 0:
            continue

        score = overlap / math.sqrt(max(1, len(query_tokens) * len(document_tokens)))
        # Exact corpus identifiers and fault-mode phrases are authoritative signals.
        if entry_id.lower() in query_lower:
            score += 0.65
        if query_asset_ids and any(asset_id in searchable.lower() for asset_id in query_asset_ids):
            score += 0.35
        fault_mode = str(entry.get("fault_mode", "")).lower()
        if fault_mode and fault_mode in query_lower:
            score += 0.35
        ranked.append((min(score, 1.0), entry))

    ranked.sort(key=lambda item: item[0], reverse=True)
    passages = [
        {
            "doc_ref": entry["id"],
            "text": entry["text"],
            "fault_mode": entry.get("fault_mode", ""),
            "relevance_score": round(score, 6),
        }
        for score, entry in ranked[:top_k]
    ]
    return {
        "retrieved_passages": passages,
        "corpus_version": "2.0.0-ironside",
        "retrieval_mode": "resource_safe_lexical",
    }
