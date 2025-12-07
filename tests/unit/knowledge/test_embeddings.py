"""
Tests for kosmos.knowledge.embeddings module.

Tests using REAL SentenceTransformer embeddings (not mocks).
Uses smaller model (all-MiniLM-L6-v2) for faster testing.
"""

import pytest
import numpy as np
import uuid

from kosmos.knowledge.embeddings import PaperEmbedder
from kosmos.literature.base_client import PaperMetadata, PaperSource


def unique_text(base: str) -> str:
    """Add unique suffix to avoid cache hits."""
    return f"{base} [test-id: {uuid.uuid4().hex[:8]}]"


@pytest.fixture
def paper_embedder():
    """Create PaperEmbedder instance with real SentenceTransformer."""
    # Use smaller model for faster tests
    embedder = PaperEmbedder(model_name="all-MiniLM-L6-v2")
    return embedder


@pytest.fixture
def specter_embedder():
    """Create PaperEmbedder with SPECTER model."""
    embedder = PaperEmbedder(model_name="allenai/specter")
    return embedder


@pytest.mark.unit
class TestPaperEmbedderInit:
    """Test paper embedder initialization."""

    def test_init_default(self):
        """Test default initialization."""
        embedder = PaperEmbedder()
        assert embedder.model_name == "allenai/specter"
        assert embedder.model is not None

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        embedder = PaperEmbedder(model_name="all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is not None


@pytest.mark.unit
class TestEmbeddingGeneration:
    """Test embedding generation."""

    def test_embed_query(self, paper_embedder):
        """Test embedding a query."""
        embedding = paper_embedder.embed_query(unique_text("test query"))

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384  # MiniLM dimension
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64

    def test_embed_paper(self, paper_embedder, sample_paper_metadata):
        """Test embedding a paper."""
        embedding = paper_embedder.embed_paper(sample_paper_metadata)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384  # MiniLM dimension

    def test_embed_papers_batch(self, paper_embedder, sample_papers_list):
        """Test batch embedding of papers."""
        embeddings = paper_embedder.embed_papers(sample_papers_list)

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == len(sample_papers_list)
        assert embeddings.shape[1] == 384  # MiniLM dimension

    def test_embed_empty_query(self, paper_embedder):
        """Test embedding empty query."""
        embedding = paper_embedder.embed_query("")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384


@pytest.mark.unit
class TestEmbeddingBehavior:
    """Test embedding behavior."""

    def test_multiple_queries_different_results(self, paper_embedder):
        """Test that different queries produce different embeddings."""
        emb1 = paper_embedder.embed_query(unique_text("machine learning"))
        emb2 = paper_embedder.embed_query(unique_text("quantum physics"))

        # Different queries should have different embeddings
        assert not np.allclose(emb1, emb2)

    def test_similar_queries_similar_embeddings(self, paper_embedder):
        """Test that similar queries produce similar embeddings."""
        emb1 = paper_embedder.embed_query("neural network deep learning")
        emb2 = paper_embedder.embed_query("deep learning neural network")

        # Similar queries should have high cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert similarity > 0.8  # High similarity


@pytest.mark.unit
class TestEmbeddingSimilarity:
    """Test similarity calculations."""

    def test_compute_similarity(self, paper_embedder):
        """Test similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])

        similarity = paper_embedder.compute_similarity(vec1, vec2)

        assert 0.99 <= similarity <= 1.01  # Should be 1.0 (identical)

    def test_compute_similarity_orthogonal(self, paper_embedder):
        """Test similarity for orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = paper_embedder.compute_similarity(vec1, vec2)

        assert -0.01 <= similarity <= 0.01  # Should be 0.0 (orthogonal)

    def test_find_most_similar(self, paper_embedder):
        """Test finding most similar papers."""
        # Create mock embeddings array
        paper_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ])

        query_embedding = np.array([1.0, 0.0, 0.0])
        similar = paper_embedder.find_most_similar(
            query_embedding, paper_embeddings, top_k=2
        )

        assert len(similar) <= 2
        assert all(isinstance(item, tuple) for item in similar)
        # First result should be most similar (index 0)
        assert similar[0][0] == 0


@pytest.mark.unit
class TestSpecterModel:
    """Test SPECTER model specifically."""

    def test_specter_embedding_dimension(self, specter_embedder):
        """Test SPECTER embedding dimension is 768."""
        embedding = specter_embedder.embed_query("test query")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768  # SPECTER dimension

    def test_specter_paper_embedding(self, specter_embedder, sample_paper_metadata):
        """Test SPECTER paper embedding."""
        embedding = specter_embedder.embed_paper(sample_paper_metadata)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768
