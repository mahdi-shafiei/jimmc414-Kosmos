"""
Tests for kosmos.agents.hypothesis_generator module.

Tests using REAL Claude API for LLM-dependent tests.
Database and literature search tests use mocks for isolation.
"""

import os
import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent
from kosmos.models.hypothesis import Hypothesis, HypothesisGenerationResponse, ExperimentType
from kosmos.literature.base_client import PaperMetadata


# Skip all tests if API key not available
pytestmark = [
    pytest.mark.requires_claude,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY for real LLM calls"
    )
]


def unique_id() -> str:
    """Generate unique ID for test isolation."""
    return uuid.uuid4().hex[:8]


@pytest.fixture
def hypothesis_agent():
    """Create HypothesisGeneratorAgent for testing."""
    return HypothesisGeneratorAgent(config={
        "model": "claude-3-haiku-20240307",
        "num_hypotheses": 3,
        "use_literature_context": False  # Disable for faster tests
    })


@pytest.mark.unit
class TestHypothesisGeneratorInit:
    """Test HypothesisGeneratorAgent initialization."""

    def test_init_default(self):
        """Test default initialization."""
        agent = HypothesisGeneratorAgent(config={
            "model": "claude-3-haiku-20240307"
        })
        assert agent.agent_type == "HypothesisGeneratorAgent"
        assert agent.num_hypotheses == 3
        assert agent.use_literature_context is True

    def test_init_with_config(self):
        """Test initialization with custom config."""
        agent = HypothesisGeneratorAgent(config={
            "model": "claude-3-haiku-20240307",
            "num_hypotheses": 5,
            "use_literature_context": False,
            "min_novelty_score": 0.7
        })
        assert agent.num_hypotheses == 5
        assert agent.use_literature_context is False
        assert agent.min_novelty_score == 0.7


@pytest.mark.unit
class TestHypothesisGeneration:
    """Test hypothesis generation with real Claude."""

    def test_generate_hypotheses_success(self, hypothesis_agent):
        """Test successful hypothesis generation with real Claude."""
        response = hypothesis_agent.generate_hypotheses(
            research_question=f"How does attention mechanism affect transformer performance? [{unique_id()}]",
            domain="machine_learning",
            store_in_db=False
        )

        # Assertions
        assert isinstance(response, HypothesisGenerationResponse)
        assert len(response.hypotheses) > 0
        assert "attention" in response.research_question.lower()
        assert response.domain == "machine_learning"
        assert response.generation_time_seconds > 0

        # Check first hypothesis
        if response.hypotheses:
            hyp = response.hypotheses[0]
            assert isinstance(hyp, Hypothesis)
            assert len(hyp.statement) > 10
            assert len(hyp.rationale) > 20

    def test_generate_with_custom_num_hypotheses(self, hypothesis_agent):
        """Test generating custom number of hypotheses."""
        response = hypothesis_agent.generate_hypotheses(
            research_question=f"How does CRISPR affect gene expression? [{unique_id()}]",
            num_hypotheses=2,
            store_in_db=False
        )

        # Should generate at least 1, up to 2
        assert len(response.hypotheses) <= 2

    def test_domain_auto_detection(self, hypothesis_agent):
        """Test automatic domain detection."""
        response = hypothesis_agent.generate_hypotheses(
            research_question=f"How do neurons communicate via synapses? [{unique_id()}]",
            domain=None,  # Auto-detect
            store_in_db=False
        )

        # Domain should be detected
        assert response.domain is not None
        assert len(response.domain) > 0


@pytest.mark.unit
class TestHypothesisValidation:
    """Test hypothesis validation."""

    def test_validate_valid_hypothesis(self, hypothesis_agent):
        """Test validating a valid hypothesis."""
        hyp = Hypothesis(
            research_question="Test question?",
            statement="Increasing parameter X will improve metric Y by 20%",
            rationale="Prior work shows that parameter X affects Y through mechanism Z. This suggests a 20% improvement.",
            domain="machine_learning"
        )

        assert hypothesis_agent._validate_hypothesis(hyp) is True

    def test_validate_statement_too_short(self, hypothesis_agent):
        """Test that Pydantic rejects hypothesis with too-short statement."""
        # Pydantic model now validates statement length during object creation
        with pytest.raises(Exception):  # ValidationError
            Hypothesis(
                research_question="Test question?",
                statement="Too short",  # Only 9 chars, min is 10
                rationale="This is a reasonable rationale with sufficient detail to explain the hypothesis.",
                domain="test"
            )

    def test_validate_rationale_too_short(self, hypothesis_agent):
        """Test that Pydantic rejects hypothesis with too-short rationale."""
        # Pydantic model now validates rationale length during object creation
        with pytest.raises(Exception):  # ValidationError
            Hypothesis(
                research_question="Test question?",
                statement="This is a reasonable hypothesis statement",
                rationale="Too short",  # Only 9 chars, min is 20
                domain="test"
            )

    def test_validate_vague_language_warning(self, hypothesis_agent, caplog):
        """Test warning for vague language (but doesn't fail)."""
        hyp = Hypothesis(
            research_question="Test question?",
            statement="Maybe increasing X might possibly improve Y somehow",
            rationale="This rationale is long enough to pass the minimum length requirement.",
            domain="test"
        )

        # Should pass but log warning
        result = hypothesis_agent._validate_hypothesis(hyp)
        assert result is True  # Doesn't fail, just warns


@pytest.mark.unit
class TestDatabaseOperations:
    """Test database storage and retrieval (uses mocks)."""

    @patch('kosmos.agents.hypothesis_generator.get_session')
    def test_store_hypothesis(self, mock_get_session, hypothesis_agent):
        """Test storing hypothesis in database."""
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=False)

        hyp = Hypothesis(
            id="test-123",
            research_question="Test question?",
            statement="Test hypothesis statement is long enough",
            rationale="Test rationale with sufficient length for validation to pass",
            domain="test_domain"
        )

        hyp_id = hypothesis_agent._store_hypothesis(hyp)

        assert hyp_id == "test-123"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch('kosmos.agents.hypothesis_generator.get_session')
    def test_get_hypothesis_by_id(self, mock_get_session, hypothesis_agent):
        """Test retrieving hypothesis by ID."""
        mock_session = MagicMock()
        mock_db_hyp = Mock()
        mock_db_hyp.id = "test-123"
        mock_db_hyp.research_question = "Test question"
        mock_db_hyp.statement = "Test statement is long enough"
        mock_db_hyp.rationale = "Test rationale is also long enough"
        mock_db_hyp.domain = "test_domain"
        mock_db_hyp.status.value = "generated"
        mock_db_hyp.testability_score = 0.8
        mock_db_hyp.novelty_score = 0.7
        mock_db_hyp.confidence_score = 0.75
        mock_db_hyp.related_papers = []
        mock_db_hyp.created_at = datetime.utcnow()
        mock_db_hyp.updated_at = datetime.utcnow()

        mock_session.query.return_value.filter.return_value.first.return_value = mock_db_hyp
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=False)

        hyp = hypothesis_agent.get_hypothesis_by_id("test-123")

        assert hyp is not None
        assert hyp.id == "test-123"
        assert "Test statement" in hyp.statement


@pytest.mark.unit
class TestLiteratureContext:
    """Test literature context gathering (uses mocks)."""

    @patch('kosmos.agents.hypothesis_generator.UnifiedLiteratureSearch')
    def test_gather_literature_context(self, mock_search_class, hypothesis_agent):
        """Test gathering literature for context."""
        from kosmos.literature.base_client import PaperSource

        # Enable literature context
        hypothesis_agent.use_literature_context = True

        mock_search = Mock()
        mock_papers = [
            PaperMetadata(
                id="paper1",
                source=PaperSource.ARXIV,
                title="Attention Is All You Need",
                authors=["Vaswani"],
                abstract="We propose the Transformer...",
                year=2017
            ),
            PaperMetadata(
                id="paper2",
                source=PaperSource.SEMANTIC_SCHOLAR,
                title="BERT",
                authors=["Devlin"],
                abstract="BERT is a transformer-based model...",
                year=2019
            )
        ]
        mock_search.search.return_value = mock_papers
        mock_search_class.return_value = mock_search
        hypothesis_agent.literature_search = mock_search

        papers = hypothesis_agent._gather_literature_context(
            research_question="How does attention work?",
            domain="machine_learning"
        )

        assert len(papers) == 2
        assert papers[0].title == "Attention Is All You Need"
        mock_search.search.assert_called_once()


@pytest.mark.unit
class TestAgentExecute:
    """Test agent execute method with real Claude."""

    def test_execute_generate_hypotheses_task(self, hypothesis_agent):
        """Test executing hypothesis generation via message."""
        from kosmos.agents.base import AgentMessage, MessageType

        message = AgentMessage(
            type=MessageType.REQUEST,
            from_agent="test",
            to_agent=hypothesis_agent.agent_id,
            content={
                "task_type": "generate_hypotheses",
                "research_question": f"How does learning rate affect training? [{unique_id()}]",
                "num_hypotheses": 2,
                "domain": "machine_learning"
            }
        )

        response = hypothesis_agent.execute(message)

        assert response.type == MessageType.RESPONSE
        assert "response" in response.content
        assert response.correlation_id == message.correlation_id


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling (uses mocks for error simulation)."""

    @patch('kosmos.agents.hypothesis_generator.get_client')
    def test_empty_llm_response(self, mock_get_client, hypothesis_agent):
        """Test handling empty LLM response."""
        mock_client = Mock()
        mock_client.generate_structured.return_value = {"hypotheses": []}
        mock_client.generate.return_value = "test"
        mock_get_client.return_value = mock_client
        hypothesis_agent.llm_client = mock_client

        response = hypothesis_agent.generate_hypotheses(
            research_question="Test?",
            store_in_db=False
        )

        assert len(response.hypotheses) == 0

    @patch('kosmos.agents.hypothesis_generator.get_client')
    def test_malformed_llm_response(self, mock_get_client, hypothesis_agent):
        """Test handling malformed LLM response."""
        mock_client = Mock()
        mock_client.generate_structured.return_value = {
            "hypotheses": [
                {"statement": "Valid statement"},  # Missing required fields
                {"rationale": "Valid rationale"}   # Missing statement
            ]
        }
        mock_client.generate.return_value = "test"
        mock_get_client.return_value = mock_client
        hypothesis_agent.llm_client = mock_client

        response = hypothesis_agent.generate_hypotheses(
            research_question="Test?",
            store_in_db=False
        )

        # Should filter out malformed hypotheses
        assert len(response.hypotheses) == 0

    @patch('kosmos.agents.hypothesis_generator.get_client')
    def test_llm_exception_handling(self, mock_get_client, hypothesis_agent):
        """Test handling LLM exceptions."""
        mock_client = Mock()
        mock_client.generate_structured.side_effect = Exception("LLM Error")
        mock_client.generate.return_value = "test"
        mock_get_client.return_value = mock_client
        hypothesis_agent.llm_client = mock_client

        response = hypothesis_agent.generate_hypotheses(
            research_question="Test?",
            store_in_db=False
        )

        # Should return empty list on error
        assert len(response.hypotheses) == 0


@pytest.mark.unit
class TestHypothesisListing:
    """Test listing hypotheses from database (uses mocks)."""

    @patch('kosmos.agents.hypothesis_generator.get_session')
    def test_list_hypotheses_all(self, mock_get_session, hypothesis_agent):
        """Test listing all hypotheses."""
        mock_session = MagicMock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.order_by.return_value.limit.return_value.all.return_value = []
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=False)

        hypotheses = hypothesis_agent.list_hypotheses(limit=100)

        assert isinstance(hypotheses, list)
        mock_session.query.assert_called_once()

    @patch('kosmos.agents.hypothesis_generator.get_session')
    def test_list_hypotheses_with_filters(self, mock_get_session, hypothesis_agent):
        """Test listing hypotheses with domain filter."""
        from kosmos.models.hypothesis import HypothesisStatus

        mock_session = MagicMock()
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value.limit.return_value.all.return_value = []
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=False)

        hypotheses = hypothesis_agent.list_hypotheses(
            domain="machine_learning",
            status=HypothesisStatus.GENERATED,
            limit=50
        )

        assert isinstance(hypotheses, list)
        # Should apply both filters
        assert mock_query.filter.call_count == 2


@pytest.mark.integration
@pytest.mark.slow
class TestHypothesisGeneratorIntegration:
    """Integration tests (require real LLM)."""

    def test_real_hypothesis_generation(self):
        """Test real hypothesis generation with Claude."""
        agent = HypothesisGeneratorAgent(config={
            "model": "claude-3-haiku-20240307",
            "num_hypotheses": 2,
            "use_literature_context": False
        })

        response = agent.generate_hypotheses(
            research_question=f"How does learning rate affect neural network convergence? [{unique_id()}]",
            domain="machine_learning",
            store_in_db=False
        )

        assert len(response.hypotheses) > 0
        assert response.domain == "machine_learning"

        for hyp in response.hypotheses:
            assert len(hyp.statement) > 15
            assert len(hyp.rationale) > 30
            assert hyp.confidence_score is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
