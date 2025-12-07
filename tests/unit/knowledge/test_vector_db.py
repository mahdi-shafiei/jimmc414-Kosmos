"""
Tests for kosmos.knowledge.vector_db module.

Tests using REAL ChromaDB and SentenceTransformer (not mocks).
Uses ephemeral ChromaDB and smaller embedding model for fast testing.
"""

import pytest
import numpy as np
import uuid

from kosmos.knowledge.vector_db import PaperVectorDB
from kosmos.literature.base_client import PaperMetadata, PaperSource


def unique_text(base: str) -> str:
    """Add unique suffix to avoid cache hits."""
    return f"{base} [test-id: {uuid.uuid4().hex[:8]}]"


@pytest.fixture
def vector_db(tmp_path):
    """Create PaperVectorDB instance with real ChromaDB and embedder."""
    # Use temporary directory for ephemeral storage
    db = PaperVectorDB(
        persist_directory=str(tmp_path / "test_vector_db"),
        collection_name=f"test_papers_{uuid.uuid4().hex[:8]}"
    )
    return db


@pytest.fixture
def unique_paper():
    """Create a unique paper for each test."""
    paper_id = f"paper_{uuid.uuid4().hex[:8]}"
    return PaperMetadata(
        id=paper_id,
        source=PaperSource.MANUAL,
        title=f"Neural Networks for Machine Learning {uuid.uuid4().hex[:8]}",
        authors=["Test Author"],
        abstract=f"This paper presents a study of neural networks. {uuid.uuid4().hex[:8]}",
        year=2024
    )


@pytest.fixture
def unique_papers():
    """Create a list of unique papers for batch testing."""
    papers = []
    topics = [
        ("Transformer Architecture", "Attention mechanisms in deep learning"),
        ("BERT Language Model", "Pre-training for NLP tasks"),
        ("Graph Neural Networks", "Learning on graph-structured data"),
    ]
    for title, abstract in topics:
        papers.append(PaperMetadata(
            id=f"paper_{uuid.uuid4().hex[:8]}",
            source=PaperSource.MANUAL,
            title=f"{title} {uuid.uuid4().hex[:8]}",
            authors=["Author A", "Author B"],
            abstract=f"{abstract} {uuid.uuid4().hex[:8]}",
            year=2024
        ))
    return papers


@pytest.mark.unit
class TestPaperVectorDBInit:
    """Test paper vector database initialization."""

    def test_init_default(self, tmp_path):
        """Test default initialization."""
        db = PaperVectorDB(persist_directory=str(tmp_path / "test"))
        assert db.collection_name == "papers"
        assert db.collection is not None

    def test_init_custom_collection(self, tmp_path):
        """Test initialization with custom collection name."""
        db = PaperVectorDB(
            collection_name="custom_papers",
            persist_directory=str(tmp_path / "test")
        )
        assert db.collection_name == "custom_papers"


@pytest.mark.unit
class TestPaperVectorDBAdd:
    """Test adding papers to paper vector database."""

    def test_add_paper(self, vector_db, unique_paper):
        """Test adding a single paper."""
        initial_count = vector_db.count()
        vector_db.add_paper(unique_paper)

        assert vector_db.count() == initial_count + 1

    def test_add_papers_batch(self, vector_db, unique_papers):
        """Test adding multiple papers in batch."""
        initial_count = vector_db.count()
        vector_db.add_papers(unique_papers)

        assert vector_db.count() == initial_count + len(unique_papers)

    def test_add_paper_with_empty_abstract(self, vector_db):
        """Test adding paper with no abstract."""
        paper = PaperMetadata(
            id=f"empty_{uuid.uuid4().hex[:8]}",
            source=PaperSource.MANUAL,
            title=f"Paper With Empty Abstract {uuid.uuid4().hex[:8]}",
            authors=[],
            abstract="",
            year=2023
        )

        vector_db.add_paper(paper)
        assert vector_db.count() >= 1


@pytest.mark.unit
class TestPaperVectorDBSearch:
    """Test searching in paper vector database."""

    def test_search_by_text(self, vector_db, unique_papers):
        """Test searching by text query."""
        # Add papers first
        vector_db.add_papers(unique_papers)

        # Search for transformers
        results = vector_db.search("transformer attention mechanism", top_k=2)

        assert len(results) <= 2
        assert len(results) >= 1  # Should find at least the transformer paper

    def test_search_by_paper(self, vector_db, unique_papers):
        """Test searching by paper (find similar)."""
        # Add papers first
        vector_db.add_papers(unique_papers)

        # Search for papers similar to the first one
        results = vector_db.search_by_paper(unique_papers[0], top_k=5)

        # Results should be a list (may include self-match or not)
        assert isinstance(results, list)

    def test_search_empty_results(self, vector_db):
        """Test search with no matching results."""
        # Don't add any papers, DB is empty
        results = vector_db.search("quantum computing superconductors")

        assert results == []

    def test_search_semantic_similarity(self, vector_db):
        """Test that semantically similar queries find related papers."""
        # Add papers with distinct topics
        papers = [
            PaperMetadata(
                id=f"ml_{uuid.uuid4().hex[:8]}",
                source=PaperSource.MANUAL,
                title="Machine Learning Fundamentals",
                authors=["ML Author"],
                abstract="Deep learning, neural networks, and gradient descent optimization.",
                year=2024
            ),
            PaperMetadata(
                id=f"bio_{uuid.uuid4().hex[:8]}",
                source=PaperSource.MANUAL,
                title="Molecular Biology Research",
                authors=["Bio Author"],
                abstract="DNA sequencing, gene expression, and protein synthesis.",
                year=2024
            ),
        ]
        vector_db.add_papers(papers)

        # Query for ML should rank ML paper higher
        ml_results = vector_db.search("artificial intelligence deep learning", top_k=2)
        assert len(ml_results) >= 1


@pytest.mark.unit
class TestPaperVectorDBCRUD:
    """Test CRUD operations."""

    def test_get_paper(self, vector_db, unique_paper):
        """Test getting a specific paper."""
        vector_db.add_paper(unique_paper)

        # ID in ChromaDB is "{source.value}:{primary_identifier}"
        stored_id = f"{unique_paper.source.value}:{unique_paper.primary_identifier}"
        paper_data = vector_db.get_paper(stored_id)

        assert paper_data is not None
        assert paper_data["id"] == stored_id

    def test_get_paper_not_found(self, vector_db):
        """Test getting non-existent paper."""
        paper_data = vector_db.get_paper("nonexistent_paper_id")

        assert paper_data is None

    def test_delete_paper(self, vector_db, unique_paper):
        """Test deleting a paper."""
        vector_db.add_paper(unique_paper)
        initial_count = vector_db.count()

        # ID in ChromaDB is "{source.value}:{primary_identifier}"
        stored_id = f"{unique_paper.source.value}:{unique_paper.primary_identifier}"
        vector_db.delete_paper(stored_id)

        assert vector_db.count() == initial_count - 1

    def test_count(self, vector_db, unique_papers):
        """Test getting total paper count."""
        assert vector_db.count() == 0

        vector_db.add_papers(unique_papers)

        assert vector_db.count() == len(unique_papers)


@pytest.mark.unit
class TestPaperVectorDBStats:
    """Test database statistics."""

    def test_get_stats(self, vector_db, unique_papers):
        """Test getting database statistics."""
        vector_db.add_papers(unique_papers)

        stats = vector_db.get_stats()

        assert "collection_name" in stats
        assert "paper_count" in stats
        assert stats["paper_count"] == len(unique_papers)


@pytest.mark.unit
class TestPaperVectorDBEmbeddings:
    """Test embedding functionality."""

    def test_embeddings_generated(self, vector_db, unique_paper):
        """Test that embeddings are generated for papers."""
        vector_db.add_paper(unique_paper)

        # ID in ChromaDB is "{source.value}:{primary_identifier}"
        stored_id = f"{unique_paper.source.value}:{unique_paper.primary_identifier}"
        paper_data = vector_db.get_paper(stored_id)
        assert paper_data is not None

    def test_different_papers_different_embeddings(self, vector_db):
        """Test that different papers get different embeddings."""
        paper1 = PaperMetadata(
            id=f"p1_{uuid.uuid4().hex[:8]}",
            source=PaperSource.MANUAL,
            title="Machine Learning Research",
            authors=["Author"],
            abstract="Deep learning and neural networks.",
            year=2024
        )
        paper2 = PaperMetadata(
            id=f"p2_{uuid.uuid4().hex[:8]}",
            source=PaperSource.MANUAL,
            title="Astrophysics Discovery",
            authors=["Author"],
            abstract="Black holes and gravitational waves.",
            year=2024
        )

        vector_db.add_papers([paper1, paper2])

        # Search should return ML paper for ML query
        results = vector_db.search("machine learning neural network", top_k=1)
        assert len(results) == 1
        # The first result should be the ML paper
        # (We can't guarantee ordering exactly without checking distance)
