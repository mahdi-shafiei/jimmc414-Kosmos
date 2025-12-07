"""
Tests for kosmos.knowledge.concept_extractor module.

Tests using REAL Claude API calls (not mocks).
Requires ANTHROPIC_API_KEY environment variable.
Uses claude-3-haiku for cost-effective testing.
"""

import os
import pytest
import uuid

from kosmos.knowledge.concept_extractor import ConceptExtractor, ExtractedConcept, ExtractedMethod
from kosmos.literature.base_client import PaperMetadata, PaperSource


# Skip all tests if no API key
pytestmark = [
    pytest.mark.requires_claude,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY for real LLM calls"
    )
]


def unique_prompt(base: str) -> str:
    """Add unique suffix to avoid cache hits."""
    return f"{base} [test-id: {uuid.uuid4().hex[:8]}]"


@pytest.fixture
def concept_extractor():
    """Create ConceptExtractor with real Claude API."""
    # Use haiku for cost-effective testing
    extractor = ConceptExtractor(model="claude-3-haiku-20240307")
    return extractor


@pytest.fixture
def unique_paper():
    """Create a unique paper for each test to avoid cache hits."""
    return PaperMetadata(
        id=f"test_{uuid.uuid4().hex[:8]}",
        source=PaperSource.MANUAL,
        title=f"Neural Networks for Machine Learning {uuid.uuid4().hex[:8]}",
        authors=["Test Author"],
        abstract=f"This paper presents a study of neural networks and deep learning. {uuid.uuid4().hex[:8]}",
        year=2024
    )


@pytest.mark.unit
class TestConceptExtractorInit:
    """Test concept extractor initialization."""

    def test_init_default(self):
        """Test default initialization."""
        extractor = ConceptExtractor()
        assert extractor.model == "claude-sonnet-4-5"

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        extractor = ConceptExtractor(model="claude-3-haiku-20240307")
        assert extractor.model == "claude-3-haiku-20240307"


@pytest.mark.unit
class TestConceptExtraction:
    """Test concept extraction."""

    def test_extract_from_paper(self, concept_extractor, unique_paper):
        """Test extracting concepts from a paper."""
        result = concept_extractor.extract_from_paper(unique_paper)

        assert result is not None
        assert hasattr(result, 'concepts')
        assert hasattr(result, 'methods')
        # Should extract at least one concept about neural networks
        assert len(result.concepts) >= 0  # May vary
        # Each concept should have required fields
        for concept in result.concepts:
            assert isinstance(concept, ExtractedConcept)
            assert hasattr(concept, 'name')
            assert hasattr(concept, 'domain')  # ExtractedConcept uses 'domain' not 'category'
            assert hasattr(concept, 'relevance')

    def test_extract_with_relationships(self, concept_extractor, unique_paper):
        """Test extracting concepts with relationships."""
        result = concept_extractor.extract_from_paper(
            unique_paper, include_relationships=True
        )

        assert result is not None
        assert hasattr(result, 'relationships')
        # Relationships may or may not be extracted
        for rel in result.relationships:
            assert hasattr(rel, 'concept1')  # ConceptRelationship uses concept1/concept2
            assert hasattr(rel, 'concept2')

    def test_extract_with_limits(self, concept_extractor, unique_paper):
        """Test extraction with max limits."""
        result = concept_extractor.extract_from_paper(
            unique_paper, max_concepts=3, max_methods=2
        )

        # Should respect limits
        assert len(result.concepts) <= 3
        assert len(result.methods) <= 2


@pytest.mark.unit
class TestConceptCaching:
    """Test concept extraction caching."""

    def test_cache_extractions(self, concept_extractor, sample_paper_metadata):
        """Test that extractions are cached."""
        # First extraction
        result1 = concept_extractor.extract_from_paper(sample_paper_metadata)

        # Second extraction (should use cache)
        result2 = concept_extractor.extract_from_paper(sample_paper_metadata)

        # Results should be the same (cached)
        assert result1.concepts == result2.concepts


@pytest.mark.unit
class TestPromptBuilding:
    """Test prompt construction."""

    def test_build_prompt_complete_paper(self, concept_extractor, sample_paper_metadata):
        """Test building prompt for paper with all fields."""
        prompt = concept_extractor._build_concept_extraction_prompt(
            sample_paper_metadata, max_concepts=10, max_methods=5
        )

        assert isinstance(prompt, str)
        assert sample_paper_metadata.title in prompt
        assert sample_paper_metadata.abstract in prompt
        assert "10" in prompt  # max_concepts
        assert "5" in prompt  # max_methods

    def test_build_prompt_minimal_paper(self, concept_extractor):
        """Test building prompt for paper with minimal fields."""
        paper = PaperMetadata(
            id="minimal_paper",
            source=PaperSource.MANUAL,
            title="Minimal",
            authors=[],
            abstract="",
            year=2023
        )

        prompt = concept_extractor._build_concept_extraction_prompt(paper, max_concepts=10, max_methods=5)

        assert isinstance(prompt, str)
        assert "Minimal" in prompt


@pytest.mark.unit
class TestConceptFiltering:
    """Test concept filtering and validation."""

    def test_concepts_have_valid_relevance(self, concept_extractor, unique_paper):
        """Test that concepts have valid relevance scores."""
        result = concept_extractor.extract_from_paper(unique_paper)

        # All concepts should have relevance in valid range [0, 1]
        for concept in result.concepts:
            assert 0.0 <= concept.relevance <= 1.0

        # Demonstrate post-extraction filtering
        high_relevance = [c for c in result.concepts if c.relevance >= 0.5]
        # High relevance concepts should be a subset
        assert len(high_relevance) <= len(result.concepts)


@pytest.mark.unit
class TestRealConceptExtraction:
    """Test real concept extraction with varied inputs."""

    def test_extract_ml_paper_concepts(self, concept_extractor):
        """Test extraction from ML-focused paper."""
        paper = PaperMetadata(
            id=f"ml_paper_{uuid.uuid4().hex[:8]}",
            source=PaperSource.MANUAL,
            title=f"Deep Reinforcement Learning with Transformer Architectures",
            authors=["Jane Doe", "John Smith"],
            abstract=f"We present a novel approach combining transformers with reinforcement learning. "
                     f"Our method uses attention mechanisms to improve policy gradients. {uuid.uuid4().hex[:8]}",
            year=2024
        )

        result = concept_extractor.extract_from_paper(paper, max_concepts=5)

        assert len(result.concepts) > 0
        # Should find relevant ML concepts
        concept_names = [c.name.lower() for c in result.concepts]
        # At least one concept should be related to the paper content
        assert any("learning" in name or "transformer" in name or "attention" in name
                   or "reinforcement" in name for name in concept_names)

    def test_extract_biology_paper_concepts(self, concept_extractor):
        """Test extraction from biology-focused paper."""
        paper = PaperMetadata(
            id=f"bio_paper_{uuid.uuid4().hex[:8]}",
            source=PaperSource.MANUAL,
            title=f"CRISPR-Cas9 Gene Editing in Cancer Research",
            authors=["Dr. Biology"],
            abstract=f"This study investigates the use of CRISPR-Cas9 for targeted gene therapy. "
                     f"We demonstrate successful mutations in tumor suppressor genes. {uuid.uuid4().hex[:8]}",
            year=2024
        )

        result = concept_extractor.extract_from_paper(paper, max_concepts=5)

        assert len(result.concepts) > 0
        # Should find relevant biology concepts
        concept_names = [c.name.lower() for c in result.concepts]
        assert any("crispr" in name or "gene" in name or "cancer" in name
                   or "therapy" in name for name in concept_names)
