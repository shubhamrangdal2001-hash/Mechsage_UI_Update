"""Diagnostics Agent — RAG boundary + strong-model reasoning.

This is the brain of the pipeline. It:
1. Maps degrading sensors to a fault hypothesis via fault_signatures.
2. Calls the RAG pipeline to retrieve relevant maintenance manual passages.
3. If RAG abstains (no relevant passage / low confidence), routes to human.
4. If passages are found, synthesises a grounded diagnosis using the PRO model.

Model: gemini-3.1-pro (strong reasoning — this is where accuracy matters most).
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import google.generativeai as genai

from dev.agentic.state import MechSageState
from dev.agentic.config import OrchestratorConfig
from dev.agentic.tools import rag_retrieve
from dev.agentic.fault_signatures import lookup_fault_hypothesis

# Load .env from project root
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

_config = OrchestratorConfig()


def _get_model() -> genai.GenerativeModel:
    """Lazy-init the strong Gemini Pro model for Diagnostics."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel(
        model_name=_config.strong_model,
        system_instruction=textwrap.dedent("""\
            You are the Diagnostics Engine for MechSage, a predictive-maintenance
            AI for Ironside Manufacturing industrial assets.

            You receive:
            - ML telemetry signals (RUL estimate, anomaly flags, degrading sensors)
            - A fault hypothesis derived from sensor patterns
            - Retrieved passages from maintenance manuals (SOPs, equipment profiles)

            Your job is to synthesise a concise, technically accurate diagnosis
            STRICTLY grounded in the retrieved passages. You must:
            1. Explain the likely root cause based on the evidence.
            2. Reference specific manual sections or document IDs.
            3. NEVER invent facts, procedures, or failure modes not in the passages.
            4. If the passages don't contain enough information, say so explicitly.

            Keep your diagnosis to 3-5 sentences. Be precise and actionable.
        """),
    )


def diagnostics_node(state: MechSageState) -> dict:
    """
    LangGraph node function for the Diagnostics Agent.

    Steps:
        1. Map degrading_sensors → fault_hypothesis via lookup table.
        2. Call rag_retrieve() with the hypothesis.
        3. If abstention → route to human (set status='abstain').
        4. If passages found → call Pro model to generate diagnosis.
        5. If confidence < cutoff → route to human.
    """
    asset_id = state["asset_id"]
    degrading_sensors = state.get("degrading_sensors", [])
    rul = state.get("rul_estimate", 0)

    # Step 1: Map sensors to fault hypothesis
    fault_hypothesis = lookup_fault_hypothesis(degrading_sensors)
    print(f"[Diagnostics] Fault hypothesis: '{fault_hypothesis}' "
          f"(from sensors: {degrading_sensors})")

    # Step 2: Call RAG pipeline
    print(f"[Diagnostics] Querying RAG pipeline for asset '{asset_id}'...")
    rag_result = rag_retrieve(
        asset_id=asset_id,
        degrading_sensors=degrading_sensors,
        fault_hypothesis=fault_hypothesis,
    )

    passages = rag_result.get("passages", [])
    confidence = rag_result.get("confidence", 0.0)
    citation = rag_result.get("citation", "")
    error_code = rag_result.get("error_code")

    # Step 3: Check for RAG abstention
    if error_code or not passages:
        reason = rag_result.get("message", "No relevant passages found.")
        msg = (
            f"[Diagnostics] ⛔ RAG ABSTAINED for '{fault_hypothesis}': "
            f"{reason}. Routing to human review."
        )
        print(msg)
        return {
            "fault_hypothesis": fault_hypothesis,
            "retrieved_passages": [],
            "diagnosis": "unresolved",
            "confidence": 0.0,
            "citation": "",
            "status": "abstain",
            "messages": [msg],
        }

    print(f"[Diagnostics] RAG returned {len(passages)} passages "
          f"(confidence: {confidence:.4f})")

    # Step 4: Check confidence threshold
    if confidence < _config.diagnosis_confidence_cutoff:
        msg = (
            f"[Diagnostics] ⚠ Low confidence ({confidence:.4f} < "
            f"{_config.diagnosis_confidence_cutoff}). Routing to human."
        )
        print(msg)
        return {
            "fault_hypothesis": fault_hypothesis,
            "retrieved_passages": passages,
            "diagnosis": "unresolved",
            "confidence": confidence,
            "citation": citation,
            "status": "abstain",
            "messages": [msg],
        }

    # Step 5: Generate diagnosis using the Pro model
    print(f"[Diagnostics] Generating diagnosis with {_config.strong_model}...")
    model = _get_model()

    context_block = "\n\n".join(
        f"[Doc {i+1} — {p.get('doc_ref', 'unknown')}]\n{p.get('text', '')}"
        for i, p in enumerate(passages)
    )

    prompt = (
        f"ASSET: {asset_id}\n"
        f"RUL ESTIMATE: {rul:.0f} cycles remaining\n"
        f"DEGRADING SENSORS: {', '.join(degrading_sensors)}\n"
        f"FAULT HYPOTHESIS: {fault_hypothesis}\n\n"
        f"RETRIEVED MAINTENANCE MANUAL PASSAGES:\n{context_block}\n\n"
        f"Based on the above evidence, provide your diagnosis. "
        f"Reference specific document IDs. Be concise (3-5 sentences)."
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 512, "temperature": 0.1},
        )
        diagnosis = response.text.strip()
    except Exception as exc:
        diagnosis = f"[DiagnosticsError] LLM generation failed: {exc}"
        print(f"[Diagnostics] ERROR: {exc}")

    msg = (
        f"[Diagnostics] ✓ Diagnosis complete (confidence: {confidence:.4f}, "
        f"model: {_config.strong_model})"
    )
    print(msg)
    print(f"[Diagnostics] Diagnosis: {diagnosis[:200]}...")

    return {
        "fault_hypothesis": fault_hypothesis,
        "retrieved_passages": passages,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "citation": citation,
        "status": "diagnosed",
        "messages": [msg],
    }
