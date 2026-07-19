"""Unit and integration tests for MechSage RAG pipeline components.

Tests cover:
- Chunking: parent/child structure, metadata propagation
- Config: path existence, field defaults (now OpenRouter)
- Retriever: BM25 tokenizer, RRF fusion logic
- Embedding: shape, normalization
- Pipeline: search happy path, abstain path, error handling
- Generator: unit (mocked) + integration (live API, requires OPENROUTER_API_KEY)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

class TestChunking:
    """Tests for dev.rag.chunking module."""

    def _sample_entry(self, entry_id="TEST-01", has_action=True):
        text = "Rising temperature in HPC stage indicates compressor degradation."
        if has_action:
            text += " Recommended action: borescope inspection and seal replacement."
        return {
            "id": entry_id,
            "fault_mode": "compressor degradation",
            "components": ["HPC", "seal"],
            "sensor_cues": ["s3", "s11"],
            "text": text,
        }

    def test_chunk_entry_returns_parent_and_children(self):
        from dev.rag.chunking import chunk_entry

        entry = self._sample_entry()
        chunks = chunk_entry(entry)
        chunk_types = {c.chunk_type for c in chunks}

        assert "parent" in chunk_types
        assert "symptom" in chunk_types
        assert "action" in chunk_types

    def test_chunk_entry_without_action_has_no_action_chunk(self):
        from dev.rag.chunking import chunk_entry

        entry = self._sample_entry(has_action=False)
        chunks = chunk_entry(entry)
        chunk_types = {c.chunk_type for c in chunks}

        assert "parent" in chunk_types
        assert "symptom" in chunk_types
        assert "action" not in chunk_types

    def test_chunk_ids_are_unique(self):
        from dev.rag.chunking import chunk_entry

        entry = self._sample_entry()
        chunks = chunk_entry(entry)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_metadata_propagated_to_children(self):
        from dev.rag.chunking import chunk_entry

        entry = self._sample_entry()
        chunks = chunk_entry(entry)
        for chunk in chunks:
            assert chunk.fault_mode == "compressor degradation"
            assert "HPC" in chunk.components
            assert "s3" in chunk.sensor_cues

    def test_parent_id_matches_entry_id(self):
        from dev.rag.chunking import chunk_entry

        entry = self._sample_entry(entry_id="MAN-XYZ-99")
        chunks = chunk_entry(entry)
        for chunk in chunks:
            assert chunk.entry_id == "MAN-XYZ-99"
            assert chunk.parent_id == "MAN-XYZ-99"

    def test_get_parent_chunks_returns_only_parents(self):
        from dev.rag.chunking import chunk_entry, get_parent_chunks

        entry = self._sample_entry()
        chunks = chunk_entry(entry)
        parent_lookup = get_parent_chunks(chunks)

        assert len(parent_lookup) == 1
        assert list(parent_lookup.values())[0].chunk_type == "parent"

    def test_chunk_knowledge_base_loads_real_kb(self):
        from dev.rag.chunking import chunk_knowledge_base

        kb_path = Path(__file__).resolve().parents[1] / "dev" / "rag" / "knowledge_base" / "knowledge_base_v2.json"
        if not kb_path.exists():
            pytest.skip("Knowledge base not found")

        chunks = chunk_knowledge_base(kb_path)
        assert len(chunks) > 0

        parent_count = sum(1 for c in chunks if c.chunk_type == "parent")
        assert parent_count > 0, "Should have at least one parent chunk"

        # Every entry should have exactly one parent chunk
        entry_ids = [c.entry_id for c in chunks if c.chunk_type == "parent"]
        assert len(entry_ids) == len(set(entry_ids)), "Duplicate parent chunks detected"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for dev.rag.config module."""

    def test_config_defaults_are_set(self):
        from dev.rag.config import RAGConfig

        config = RAGConfig()
        assert config.embedding_provider == "local"
        assert config.min_relevance_score == 0.42, (
            "min_relevance_score was raised from 0.40 to 0.42 after near-miss audit — "
            "keep this test in sync with config.py"
        )
        assert config.reranker_top_k == 3
        assert config.dense_top_k == 10
        assert config.bm25_top_k == 10
        # LLM provider is now OpenRouter
        assert config.llm_provider == "openrouter"
        assert hasattr(config, "openrouter_model")
        assert hasattr(config, "openrouter_base_url")

    def test_abstain_queries_path_attribute_exists(self):
        from dev.rag.config import RAGConfig

        config = RAGConfig()
        assert hasattr(config, "abstain_queries_path"), (
            "abstain_queries_path must be a config field, not hardcoded in evaluate_rag.py"
        )
        assert "abstain_queries" in str(config.abstain_queries_path)

    def test_kb_path_attribute_exists(self):
        from dev.rag.config import RAGConfig

        config = RAGConfig()
        assert config.kb_path.name == "knowledge_base_v2.json"

    def test_test_queries_path_attribute_exists(self):
        from dev.rag.config import RAGConfig

        config = RAGConfig()
        assert config.test_queries_path.name == "test_queries.json"

    def test_min_reranker_score_field_exists(self):
        """Ensure min_reranker_score config field is present for the reranker threshold."""
        from dev.rag.config import RAGConfig

        config = RAGConfig()
        assert hasattr(config, "min_reranker_score"), (
            "min_reranker_score must exist in RAGConfig to support reranker score floor"
        )
        assert config.min_reranker_score <= 0, (
            "min_reranker_score should be <= 0 (cross-encoder logits for irrelevant pairs are negative)"
        )


# ---------------------------------------------------------------------------
# Retriever helpers tests
# ---------------------------------------------------------------------------

class TestRetrieverHelpers:
    """Tests for tokenizer and RRF logic in dev.rag.retriever."""

    def test_tokenize_lowercases_and_splits(self):
        from dev.rag.retriever import tokenize

        result = tokenize("Rising HPC-Outlet Temperature: s3, s7!")
        assert "rising" in result
        assert "hpc" in result
        assert "outlet" in result
        assert "temperature" in result
        assert "s3" in result
        assert "s7" in result

    def test_tokenize_empty_string(self):
        from dev.rag.retriever import tokenize

        assert tokenize("") == []

    def test_rrf_scores_merge_correctly(self):
        """RRF: a chunk appearing in both dense and sparse lists should score higher."""
        from dev.rag.config import RAGConfig
        from dev.rag.retriever import HybridRetriever
        from dev.rag.chunking import Chunk

        # Build minimal mocks
        config = RAGConfig()
        config.rrf_k = 60

        chunk_a = Chunk(
            chunk_id="A__parent", text="HPC degradation rising temperature",
            chunk_type="parent", parent_id="A", entry_id="A",
            fault_mode="hpc", components=[], sensor_cues=[]
        )
        chunk_b = Chunk(
            chunk_id="B__parent", text="Fan imbalance vibration",
            chunk_type="parent", parent_id="B", entry_id="B",
            fault_mode="fan", components=[], sensor_cues=[]
        )

        mock_collection = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = np.zeros(384)

        retriever = HybridRetriever(
            config=config,
            collection=mock_collection,
            embed_model=mock_embed,
            chunks=[chunk_a, chunk_b]
        )

        # Chunk A appears in both dense and sparse → should have a higher RRF score
        dense = [
            {"chunk_id": "A__parent", "text": chunk_a.text, "metadata": {}, "raw_score": 0.1, "retriever": "dense"},
            {"chunk_id": "B__parent", "text": chunk_b.text, "metadata": {}, "raw_score": 0.2, "retriever": "dense"},
        ]
        sparse = [
            {"chunk_id": "A__parent", "text": chunk_a.text, "metadata": {}, "raw_score": 5.0, "retriever": "sparse"},
        ]

        fused = retriever.reciprocal_rank_fusion(dense, sparse)
        assert fused[0]["chunk_id"] == "A__parent", "Chunk A must rank first after RRF fusion"


# ---------------------------------------------------------------------------
# NDCG helper tests
# ---------------------------------------------------------------------------

class TestNDCGHelper:
    """Tests for calculate_ndcg in evaluate_rag."""

    def test_ndcg_rank_1_at_k1(self):
        from dev.rag.evaluate_rag import calculate_ndcg
        assert abs(calculate_ndcg(1, 1) - 1.0) < 1e-6

    def test_ndcg_rank_1_at_k3(self):
        from dev.rag.evaluate_rag import calculate_ndcg
        expected = 1.0 / math.log2(2)
        assert abs(calculate_ndcg(1, 3) - expected) < 1e-6

    def test_ndcg_rank_2_at_k3(self):
        from dev.rag.evaluate_rag import calculate_ndcg
        expected = 1.0 / math.log2(3)
        assert abs(calculate_ndcg(2, 3) - expected) < 1e-6

    def test_ndcg_rank_0_returns_0(self):
        from dev.rag.evaluate_rag import calculate_ndcg
        assert calculate_ndcg(0, 3) == 0.0

    def test_ndcg_rank_beyond_k_returns_0(self):
        from dev.rag.evaluate_rag import calculate_ndcg
        assert calculate_ndcg(4, 3) == 0.0


# ---------------------------------------------------------------------------
# Context Precision / Recall helper tests
# ---------------------------------------------------------------------------

class TestRAGASHelpers:
    """Tests for RAGAS-style context metric helpers in evaluate_rag."""

    def test_context_precision_rank_1(self):
        from dev.rag.evaluate_rag import context_precision_at_k
        assert context_precision_at_k(1, 3) == 1.0

    def test_context_precision_rank_0(self):
        from dev.rag.evaluate_rag import context_precision_at_k
        assert context_precision_at_k(0, 3) == 0.0

    def test_context_precision_beyond_k(self):
        from dev.rag.evaluate_rag import context_precision_at_k
        assert context_precision_at_k(5, 3) == 0.0

    def test_context_recall_rank_within_k(self):
        from dev.rag.evaluate_rag import context_recall_at_k
        assert context_recall_at_k(3, 3) == 1.0

    def test_context_recall_rank_0(self):
        from dev.rag.evaluate_rag import context_recall_at_k
        assert context_recall_at_k(0, 3) == 0.0

    def test_context_recall_beyond_k(self):
        from dev.rag.evaluate_rag import context_recall_at_k
        assert context_recall_at_k(4, 3) == 0.0


# ---------------------------------------------------------------------------
# Pipeline singleton test
# ---------------------------------------------------------------------------

class TestPipelineSingleton:
    """Tests for the module-level singleton in rag_pipeline."""

    def test_get_pipeline_returns_same_instance(self):
        from dev.rag.rag_pipeline import get_pipeline

        # Mock to avoid loading real models
        with patch("dev.rag.rag_pipeline.MechSageRAGPipeline") as MockPipeline:
            instance = MagicMock()
            MockPipeline.return_value = instance

            # Reset singleton
            import dev.rag.rag_pipeline as rp
            rp._PIPELINE_SINGLETON = None

            p1 = get_pipeline()
            p2 = get_pipeline()

            # Second call should NOT create a new instance
            assert MockPipeline.call_count == 1, (
                "MechSageRAGPipeline should only be instantiated once (singleton pattern)"
            )
            assert p1 is p2

    def test_rebuild_index_forces_new_instance(self):
        from dev.rag.rag_pipeline import get_pipeline

        with patch("dev.rag.rag_pipeline.MechSageRAGPipeline") as MockPipeline:
            import dev.rag.rag_pipeline as rp
            rp._PIPELINE_SINGLETON = None

            get_pipeline()
            get_pipeline(rebuild_index=True)

            assert MockPipeline.call_count == 2, (
                "rebuild_index=True must create a new pipeline instance"
            )


# ---------------------------------------------------------------------------
# Embedding shape tests
# ---------------------------------------------------------------------------

class TestEmbeddingShape:
    """Tests for edge cases in dev.rag.embedding.EmbeddingModel."""

    @pytest.fixture(autouse=True)
    def deterministic_sentence_transformer(self, monkeypatch):
        """Avoid network/model-cache dependence while exercising wrapper logic."""
        import numpy as np
        import sentence_transformers

        class FakeSentenceTransformer:
            def __init__(self, model_name):
                self.model_name = model_name

            def get_sentence_embedding_dimension(self):
                return 4

            def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
                vectors = np.array(
                    [[len(text), 1.0, 2.0, 3.0] for text in texts], dtype=np.float32
                )
                if normalize_embeddings:
                    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                return vectors

        monkeypatch.setattr(
            sentence_transformers, "SentenceTransformer", FakeSentenceTransformer
        )

    def test_embed_empty_list_returns_correct_shape(self):
        """embed([]) must return shape (0, dim) not (0,) — downstream code expects 2-D arrays."""
        from dev.rag.embedding import EmbeddingModel

        model = EmbeddingModel(provider="local")
        result = model.embed([])
        assert result.ndim == 2, f"Expected 2-D array, got shape {result.shape}"
        assert result.shape[0] == 0
        assert result.shape[1] == model.dimension

    def test_embed_query_returns_1d_vector(self):
        """embed_query must return a 1-D vector of correct dimension."""
        from dev.rag.embedding import EmbeddingModel

        model = EmbeddingModel(provider="local")
        vec = model.embed_query("HPC temperature anomaly")
        assert vec.ndim == 1
        assert vec.shape[0] == model.dimension

    def test_local_embeddings_are_l2_normalized(self):
        """Local model embeddings must be unit vectors so dot product == cosine sim."""
        from dev.rag.embedding import EmbeddingModel
        import numpy as np

        model = EmbeddingModel(provider="local")
        vec = model.embed_query("bearing wear detected")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm:.6f}"


# ---------------------------------------------------------------------------
# Concurrent singleton test
# ---------------------------------------------------------------------------

class TestSingletonConcurrency:
    """Tests for thread safety of get_pipeline singleton."""

    def test_concurrent_get_pipeline_creates_only_one_instance(self):
        """Concurrent calls to get_pipeline() must not create duplicate instances."""
        import threading
        from dev.rag.rag_pipeline import get_pipeline

        with patch("dev.rag.rag_pipeline.MechSageRAGPipeline") as MockPipeline:
            import dev.rag.rag_pipeline as rp
            rp._PIPELINE_SINGLETON = None
            MockPipeline.return_value = MagicMock()

            instances = []
            errors = []

            def call_get_pipeline():
                try:
                    instances.append(get_pipeline())
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=call_get_pipeline) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Threads raised exceptions: {errors}"
            # All 10 threads should get the same singleton
            assert len(set(id(i) for i in instances)) == 1, (
                "Multiple instances were created — singleton is not thread-safe"
            )
            assert MockPipeline.call_count == 1, (
                f"MechSageRAGPipeline created {MockPipeline.call_count} times instead of 1"
            )


# ---------------------------------------------------------------------------
# asset_id warning test
# ---------------------------------------------------------------------------

class TestManualRetrievalRAG:
    """Tests for the manual_retrieval_rag entry point."""

    def test_asset_id_logs_warning_when_provided(self, caplog):
        """asset_id must not be silently ignored — it must log a warning."""
        import logging
        from dev.rag.rag_pipeline import manual_retrieval_rag

        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = {"retrieved_passages": []}

        with patch("dev.rag.rag_pipeline.get_pipeline", return_value=mock_pipeline):
            with caplog.at_level(logging.WARNING, logger="dev.rag.rag_pipeline"):
                manual_retrieval_rag(
                    query="HPC temperature rising",
                    asset_id="CFM56-7B"
                )

        assert any("asset_id" in r.message for r in caplog.records), (
            "Expected a warning log mentioning asset_id, but none was emitted"
        )


# ---------------------------------------------------------------------------
# Reranker score floor test
# ---------------------------------------------------------------------------

class TestRerankerScoreFloor:
    """Tests for the reranker min_reranker_score threshold."""

    def test_reranker_filters_below_floor_score(self):
        """Candidates scoring below min_reranker_score must be excluded from output."""
        from dev.rag.config import RAGConfig
        from dev.rag.reranker import CrossEncoderReranker
        from dev.rag.chunking import Chunk

        config = RAGConfig()
        config.min_reranker_score = 0.0  # strict floor for this test
        config.reranker_top_k = 5

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.config = config

        # Simulate cross-encoder scores: one clearly negative
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([5.0, -3.0, 2.0])
        reranker.model = mock_model

        candidates = [
            {"chunk_id": "A", "text": "good"},
            {"chunk_id": "B", "text": "garbage"},
            {"chunk_id": "C", "text": "decent"},
        ]

        result = reranker.rerank("query", candidates)
        returned_ids = [r["chunk_id"] for r in result]

        assert "B" not in returned_ids, "Candidate with score -3.0 should be filtered out"
        assert "A" in returned_ids
        assert "C" in returned_ids
        assert result[0]["chunk_id"] == "A", "Highest scoring candidate must rank first"

    def test_reranker_returns_empty_when_all_below_floor(self):
        """If all candidates score below floor, reranker should return empty list."""
        from dev.rag.config import RAGConfig
        from dev.rag.reranker import CrossEncoderReranker

        config = RAGConfig()
        config.min_reranker_score = 0.0
        config.reranker_top_k = 3

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.config = config
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-5.0, -8.0])
        reranker.model = mock_model

        candidates = [
            {"chunk_id": "X", "text": "irrelevant"},
            {"chunk_id": "Y", "text": "also irrelevant"},
        ]

        result = reranker.rerank("query", candidates)
        assert result == [], "Expected empty list when all candidates are below floor"


# ---------------------------------------------------------------------------
# KB schema validation tests
# ---------------------------------------------------------------------------

class TestKBValidation:
    """Tests for knowledge base schema validation in dev.rag.chunking."""

    def test_missing_id_raises_value_error(self, tmp_path):
        """KB entry missing 'id' field must raise ValueError with clear message."""
        import json
        from dev.rag.chunking import chunk_knowledge_base

        bad_kb = [{"text": "some text", "fault_mode": "degradation"}]  # missing 'id'
        kb_file = tmp_path / "bad_kb.json"
        kb_file.write_text(json.dumps(bad_kb), encoding="utf-8")

        with pytest.raises(ValueError, match="id"):
            chunk_knowledge_base(kb_file)

    def test_missing_text_raises_value_error(self, tmp_path):
        """KB entry missing 'text' field must raise ValueError with clear message."""
        import json
        from dev.rag.chunking import chunk_knowledge_base

        bad_kb = [{"id": "TEST-01", "fault_mode": "degradation"}]  # missing 'text'
        kb_file = tmp_path / "bad_kb.json"
        kb_file.write_text(json.dumps(bad_kb), encoding="utf-8")

        with pytest.raises(ValueError, match="text"):
            chunk_knowledge_base(kb_file)

    def test_missing_kb_file_raises_file_not_found(self, tmp_path):
        """Pointing to a non-existent KB file must raise FileNotFoundError."""
        from dev.rag.chunking import chunk_knowledge_base

        missing_path = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match="Knowledge base file not found"):
            chunk_knowledge_base(missing_path)

    def test_valid_entry_with_optional_fields_is_accepted(self, tmp_path):
        """A valid KB entry (required + optional fields) must not raise any exception."""
        import json
        from dev.rag.chunking import chunk_knowledge_base

        valid_kb = [{
            "id": "TEST-01",
            "fault_mode": "compressor degradation",
            "text": "Rising HPC temperature. Recommended action: borescope inspection.",
            "components": ["HPC"],
            "sensor_cues": ["s3"]
        }]
        kb_file = tmp_path / "valid_kb.json"
        kb_file.write_text(json.dumps(valid_kb), encoding="utf-8")

        chunks = chunk_knowledge_base(kb_file)
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# New Evaluation Metrics tests
# ---------------------------------------------------------------------------

class TestNewEvalMetrics:
    """Tests for new RAG metrics added in evaluate_rag."""

    def test_average_precision_at_k_rank_1(self):
        from dev.rag.evaluate_rag import average_precision_at_k
        assert average_precision_at_k(1, 3) == 1.0

    def test_average_precision_at_k_rank_2(self):
        from dev.rag.evaluate_rag import average_precision_at_k
        assert average_precision_at_k(2, 3) == 0.5

    def test_average_precision_at_k_rank_3(self):
        from dev.rag.evaluate_rag import average_precision_at_k
        assert average_precision_at_k(3, 3) == 1.0 / 3.0

    def test_average_precision_at_k_out_of_bounds(self):
        from dev.rag.evaluate_rag import average_precision_at_k
        assert average_precision_at_k(4, 3) == 0.0
        assert average_precision_at_k(0, 3) == 0.0

    def test_check_retriever_success_found(self):
        from dev.rag.evaluate_rag import check_retriever_success
        results = [
            {"metadata": {"entry_id": "MAN-HPC-01"}},
            {"metadata": {"entry_id": "MAN-FAN-02"}},
        ]
        assert check_retriever_success(results, "MAN-FAN-02") is True

    def test_check_retriever_success_not_found(self):
        from dev.rag.evaluate_rag import check_retriever_success
        results = [
            {"metadata": {"entry_id": "MAN-HPC-01"}},
            {"metadata": {"entry_id": "MAN-FAN-02"}},
        ]
        assert check_retriever_success(results, "MAN-BRG-01") is False


class TestAssetScopedFiltering:
    """Tests to verify metadata-based asset-scoped RAG filtering."""

    def test_chunking_assigns_correct_asset_types(self):
        from dev.rag.chunking import chunk_entry
        
        # Test C-MAPSS entry
        entry_hpc = {"id": "MAN-HPC-01", "fault_mode": "hpc", "text": "HPC degradation. Recommended action: inspect."}
        chunks_hpc = chunk_entry(entry_hpc)
        assert chunks_hpc[0].asset_type == "turbofan"
        
        # Test AI4I entry
        entry_twf = {"id": "MAN-TWF-01", "fault_mode": "twf", "text": "Tool wear failure. Recommended action: change."}
        chunks_twf = chunk_entry(entry_twf)
        assert chunks_twf[0].asset_type == "milling_machine"
        
        # Test general guide entry
        entry_guide = {"id": "MAN-GUIDE-01", "fault_mode": "general", "text": "General maintenance guide."}
        chunks_guide = chunk_entry(entry_guide)
        assert chunks_guide[0].asset_type == "general"

        # Test Ironside entries via prefix
        entry_ism = {"id": "ISM-CNC-001", "fault_mode": "bearing", "text": "CNC spindle bearing wear."}
        chunks_ism = chunk_entry(entry_ism)
        assert chunks_ism[0].asset_type == "ironside"

        entry_sop = {"id": "SOP-PM-001", "fault_mode": "general", "text": "Bearing replacement SOP."}
        chunks_sop = chunk_entry(entry_sop)
        assert chunks_sop[0].asset_type == "ironside"

        # Test Ironside entry via explicit metadata field
        entry_explicit = {"id": "TEST-EXPL-01", "fault_mode": "general", "text": "Explicit test.", "asset_type": "ironside"}
        chunks_explicit = chunk_entry(entry_explicit)
        assert chunks_explicit[0].asset_type == "ironside"

    @patch("dev.rag.rag_pipeline.get_pipeline")
    def test_manual_retrieval_rag_derives_correct_filters(self, mock_get_pipeline):
        from dev.rag.rag_pipeline import manual_retrieval_rag
        
        mock_pipeline_instance = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline_instance
        
        # Test C-MAPSS engine ID mapping to turbofan filter
        manual_retrieval_rag("query text", asset_id="engine-077")
        mock_pipeline_instance.search.assert_called_with(
            query="query text",
            degrading_sensors=None,
            fault_hypothesis="",
            top_k=3,
            asset_filter="turbofan"
        )
        
        # Test AI4I machine ID mapping to milling_machine filter
        manual_retrieval_rag("query text", asset_id="milling-machine-01")
        mock_pipeline_instance.search.assert_called_with(
            query="query text",
            degrading_sensors=None,
            fault_hypothesis="",
            top_k=3,
            asset_filter="milling_machine"
        )

        # Test Ironside asset ID mapping to ironside filter
        manual_retrieval_rag("query text", asset_id="ISM-CNC-001")
        mock_pipeline_instance.search.assert_called_with(
            query="query text",
            degrading_sensors=None,
            fault_hypothesis="",
            top_k=3,
            asset_filter="ironside"
        )

        manual_retrieval_rag("query text", asset_id="wo-2025-0012")
        mock_pipeline_instance.search.assert_called_with(
            query="query text",
            degrading_sensors=None,
            fault_hypothesis="",
            top_k=3,
            asset_filter="ironside"
        )
        
        # Test missing / generic asset ID does not apply filters
        manual_retrieval_rag("query text", asset_id="some-unknown-id")
        mock_pipeline_instance.search.assert_called_with(
            query="query text",
            degrading_sensors=None,
            fault_hypothesis="",
            top_k=3,
            asset_filter=None
        )


# ---------------------------------------------------------------------------
# Generator unit tests (mocked — no real API call)
# ---------------------------------------------------------------------------

class TestGeneratorUnit:
    """Unit tests for MechSageGenerator using a mocked OpenAI client."""

    def _make_mock_response(self, content: str):
        """Build a minimal mock that looks like an openai ChatCompletion response."""
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_generate_answer_returns_string(self):
        """generate_answer must return a non-empty string when passages are provided."""
        from dev.rag.generator import MechSageGenerator

        gen = MechSageGenerator.__new__(MechSageGenerator)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Borescope inspection is recommended."
        )
        gen._client = mock_client
        gen._model_name = "anthropic/claude-sonnet-4-5"

        passages = [{"text": "HPC blades show erosion.", "doc_ref": "MAN-HPC-01", "fault_mode": "hpc"}]
        result = gen.generate_answer("What is the fault?", passages)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "[GenerationError]" not in result

    def test_generate_answer_no_passages_returns_fallback(self):
        """generate_answer with empty passages must return the 'no context' fallback."""
        from dev.rag.generator import MechSageGenerator

        gen = MechSageGenerator.__new__(MechSageGenerator)
        result = gen.generate_answer("What is the fault?", [])
        assert result == "No relevant context was retrieved."

    def test_retry_on_429_then_succeeds(self):
        """_call_with_retry must retry on 429 errors and return success on later attempt."""
        from dev.rag.generator import MechSageGenerator

        gen = MechSageGenerator.__new__(MechSageGenerator)
        mock_client = MagicMock()
        gen._client = mock_client
        gen._model_name = "anthropic/claude-sonnet-4-5"

        # First call raises 429, second call succeeds
        good_response = self._make_mock_response("Success after retry.")
        mock_client.chat.completions.create.side_effect = [
            Exception("429 Too Many Requests"),
            good_response,
        ]

        result = gen._call_with_retry(
            [{"role": "user", "content": "test"}],
            max_tokens=128,
        )
        assert result == "Success after retry."
        assert mock_client.chat.completions.create.call_count == 2

    def test_require_api_key_raises_when_missing(self):
        """_require_api_key must raise EnvironmentError when OPENROUTER_API_KEY is unset."""
        import os
        from dev.rag.generator import _require_api_key

        original = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
                _require_api_key()
        finally:
            if original is not None:
                os.environ["OPENROUTER_API_KEY"] = original

    def test_judge_faithfulness_parses_response(self):
        """judge_faithfulness must correctly parse TOTAL_CLAIMS/SUPPORTED_CLAIMS lines."""
        from dev.rag.generator import MechSageGenerator

        gen = MechSageGenerator.__new__(MechSageGenerator)
        mock_client = MagicMock()
        gen._client = mock_client
        gen._model_name = "anthropic/claude-sonnet-4-5"

        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "TOTAL_CLAIMS: 3\nSUPPORTED_CLAIMS: 2\nREASONING: One claim lacked direct support."
        )

        result = gen.judge_faithfulness("answer text", [{"text": "context"}])
        assert result["total"] == 3
        assert result["supported"] == 2
        assert abs(result["score"] - 2 / 3) < 1e-6

    def test_judge_answer_relevancy_parses_score(self):
        """judge_answer_relevancy must extract the SCORE float correctly."""
        from dev.rag.generator import MechSageGenerator

        gen = MechSageGenerator.__new__(MechSageGenerator)
        mock_client = MagicMock()
        gen._client = mock_client
        gen._model_name = "anthropic/claude-sonnet-4-5"

        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "SCORE: 0.92\nREASONING: The answer directly addresses the question."
        )

        result = gen.judge_answer_relevancy("query", "answer")
        assert abs(result["score"] - 0.92) < 1e-6

    def test_judge_explanation_quality_fills_missing_dims(self):
        """judge_explanation_quality must fill missing dimension scores with 3.0."""
        from dev.rag.generator import MechSageGenerator

        gen = MechSageGenerator.__new__(MechSageGenerator)
        mock_client = MagicMock()
        gen._client = mock_client
        gen._model_name = "anthropic/claude-sonnet-4-5"

        # Only CLARITY returned — rest should default to 3.0
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "CLARITY: 5"
        )

        result = gen.judge_explanation_quality("query", "answer", "context")
        assert result["clarity"] == 5.0
        assert result["actionability"] == 3.0
        assert result["technical_accuracy"] == 3.0
        assert result["conciseness"] == 3.0
        assert "overall" in result
        assert "overall_normalized" in result


# ---------------------------------------------------------------------------
# Generator integration tests (live API — skipped when key is absent)
# ---------------------------------------------------------------------------

def _openrouter_key_available() -> bool:
    """Return True if OPENROUTER_API_KEY is set (non-empty) in environment."""
    import os
    from pathlib import Path
    # Try loading .env first so CI/local both work
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    except ImportError:
        pass
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


@pytest.mark.skipif(
    not _openrouter_key_available(),
    reason="OPENROUTER_API_KEY not set — skipping live API integration tests"
)
class TestGeneratorIntegration:
    """Live integration tests against OpenRouter API.

    These tests make real HTTP calls and consume API quota.
    They are skipped automatically when OPENROUTER_API_KEY is absent.
    Run with: pytest -m 'not skipif' or just set the env var.
    """

    def test_generate_answer_live(self):
        """Live smoke test: generate_answer returns a non-empty string from the API."""
        from dev.rag.generator import MechSageGenerator

        gen = MechSageGenerator()  # reads key from env / .env
        passages = [{
            "text": (
                "High Pressure Compressor (HPC) degradation is indicated by rising "
                "EGT and declining N2 speed. Recommended action: borescope inspection "
                "of HPC blades and replacement of worn seals."
            ),
            "doc_ref": "MAN-HPC-01",
            "fault_mode": "hpc_degradation",
        }]
        query = "What are the signs of HPC degradation and what should a technician do?"
        answer = gen.generate_answer(query, passages)

        assert isinstance(answer, str), "Answer must be a string"
        assert len(answer) > 20, "Answer appears too short — possible API error"
        assert "[GenerationError]" not in answer, f"Generation failed: {answer}"
        assert "[RateLimitError]" not in answer, f"Rate limited: {answer}"
        print(f"\n[Integration] generate_answer response:\n{answer}")

    def test_api_key_validity(self):
        """Verify the OPENROUTER_API_KEY works by making a minimal API call."""
        from dev.rag.generator import MechSageGenerator

        gen = MechSageGenerator()
        # Minimal 1-token prompt to check auth
        response = gen._call_with_retry(
            [{"role": "user", "content": "Reply with just: OK"}],
            max_tokens=10,
            temperature=0.0,
        )
        assert response, "Empty response — possible auth failure"
        assert "[RateLimitError]" not in response
        print(f"\n[Integration] API key check response: {response!r}")

    def test_judge_faithfulness_live(self):
        """Live test: judge_faithfulness returns a valid score dict from the API."""
        from dev.rag.generator import MechSageGenerator

        gen = MechSageGenerator()
        passages = [{"text": "Bearing wear causes elevated vibration and noise."}]
        answer = "Bearing wear leads to increased vibration."

        result = gen.judge_faithfulness(answer, passages)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert "reasoning" in result
        print(f"\n[Integration] Faithfulness result: {result}")
