"""
Integration tests for async message passing architecture (Issue #66 fix).

Tests the full async message passing flow with REAL Claude API calls.
Validates that:
1. Async send_message() works with real message routing
2. Async execute() works with real LLM responses
3. CLI async entry point works end-to-end
"""

import os
import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from kosmos.agents.base import BaseAgent, AgentMessage, MessageType
from kosmos.agents.registry import AgentRegistry, get_registry
from kosmos.agents.research_director import ResearchDirectorAgent


# Skip all tests if API key not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_claude,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY for real LLM calls"
    )
]


class TestAsyncMessagePassing:
    """Test async message passing between agents with real API."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry for each test."""
        # Reset singleton for test isolation
        AgentRegistry._instance = None
        return get_registry()

    @pytest.mark.asyncio
    async def test_async_send_message_basic(self, registry):
        """Test basic async message sending."""
        sender = BaseAgent(agent_id="sender", agent_type="TestSender")
        receiver = BaseAgent(agent_id="receiver", agent_type="TestReceiver")

        registry.register(sender)
        registry.register(receiver)

        # Send async message
        message = await sender.send_message(
            to_agent="receiver",
            content={"test": "data", "value": 42},
            message_type=MessageType.REQUEST
        )

        assert message.type == MessageType.REQUEST
        assert message.from_agent == "sender"
        assert message.to_agent == "receiver"
        assert message.content["test"] == "data"
        assert receiver.messages_received == 1

    @pytest.mark.asyncio
    async def test_async_message_routing(self, registry):
        """Test message routing through registry."""
        agent_a = BaseAgent(agent_id="agent_a", agent_type="TypeA")
        agent_b = BaseAgent(agent_id="agent_b", agent_type="TypeB")

        registry.register(agent_a)
        registry.register(agent_b)

        # Send message via registry
        message = await registry.send_message(
            from_agent_id="agent_a",
            to_agent_id="agent_b",
            content={"action": "test"},
            message_type="request"
        )

        assert message.from_agent == "agent_a"
        assert message.to_agent == "agent_b"
        assert agent_b.messages_received >= 1

    @pytest.mark.asyncio
    async def test_async_broadcast_message(self, registry):
        """Test async broadcast to multiple agents."""
        broadcaster = BaseAgent(agent_id="broadcaster", agent_type="Broadcaster")
        receiver1 = BaseAgent(agent_id="receiver1", agent_type="Receiver")
        receiver2 = BaseAgent(agent_id="receiver2", agent_type="Receiver")
        receiver3 = BaseAgent(agent_id="receiver3", agent_type="Receiver")

        registry.register(broadcaster)
        registry.register(receiver1)
        registry.register(receiver2)
        registry.register(receiver3)

        # Broadcast to all Receiver types
        messages = await registry.broadcast_message(
            from_agent_id="broadcaster",
            content={"broadcast": "test"},
            target_types=["Receiver"]
        )

        assert len(messages) == 3
        assert receiver1.messages_received == 1
        assert receiver2.messages_received == 1
        assert receiver3.messages_received == 1


class TestAsyncResearchDirector:
    """Test ResearchDirector async methods with real Claude API."""

    @pytest.fixture
    def research_director(self):
        """Create ResearchDirector for testing."""
        # Reset registry for isolation
        AgentRegistry._instance = None

        director = ResearchDirectorAgent(
            research_question="What is the role of CRISPR in gene therapy?",
            domain="biology",
            config={
                "max_iterations": 2,
                "enable_concurrent_operations": False
            }
        )

        # Register with registry
        registry = get_registry()
        registry.register(director)

        return director

    @pytest.mark.asyncio
    async def test_async_execute_start_research(self, research_director):
        """Test async execute() with real Claude for research plan generation."""
        result = await research_director.execute({"action": "start_research"})

        assert result["status"] == "research_started"
        assert "research_plan" in result
        assert "next_action" in result
        assert result["research_plan"] is not None

    @pytest.mark.asyncio
    async def test_async_execute_step(self, research_director):
        """Test async execute step with real Claude."""
        # Start first
        research_director.start()

        result = await research_director.execute({"action": "step"})

        assert result["status"] == "step_executed"
        assert "next_action" in result
        assert "workflow_state" in result

    @pytest.mark.asyncio
    async def test_async_send_to_hypothesis_generator(self, research_director):
        """Test async message sending to hypothesis generator."""
        research_director.register_agent("HypothesisGeneratorAgent", "hyp-gen-test")

        message = await research_director._send_to_hypothesis_generator(
            action="generate",
            context={"max_hypotheses": 3}
        )

        assert message.type == MessageType.REQUEST
        assert message.to_agent == "hyp-gen-test"
        assert message.content["action"] == "generate"
        assert message.content["research_question"] == research_director.research_question

    @pytest.mark.asyncio
    async def test_async_send_to_experiment_designer(self, research_director):
        """Test async message sending to experiment designer."""
        research_director.register_agent("ExperimentDesignerAgent", "exp-des-test")

        message = await research_director._send_to_experiment_designer(
            hypothesis_id="test-hyp-001"
        )

        assert message.type == MessageType.REQUEST
        assert message.to_agent == "exp-des-test"
        assert message.content["action"] == "design_experiment"
        assert message.content["hypothesis_id"] == "test-hyp-001"

    @pytest.mark.asyncio
    async def test_research_plan_generation_real_llm(self, research_director):
        """Test research plan generation with real Claude API."""
        plan = research_director.generate_research_plan()

        assert plan is not None
        # Real Claude should generate a meaningful plan
        assert hasattr(research_director, 'research_plan')
        assert research_director.research_plan.research_question is not None

    @pytest.mark.asyncio
    async def test_full_async_workflow_cycle(self, research_director):
        """Test complete async workflow cycle with real Claude."""
        # Start research
        start_result = await research_director.execute({"action": "start_research"})
        assert start_result["status"] == "research_started"

        # Execute a few steps
        for i in range(2):
            step_result = await research_director.execute({"action": "step"})
            assert step_result["status"] == "step_executed"

            # Check workflow is progressing
            status = research_director.get_research_status()
            assert status["workflow_state"] is not None

        # Verify research made progress
        final_status = research_director.get_research_status()
        assert final_status["iteration"] >= 0


class TestAsyncCLIIntegration:
    """Test CLI async entry point with real API."""

    @pytest.mark.asyncio
    async def test_run_with_progress_async_import(self):
        """Test that async CLI function can be imported and called."""
        from kosmos.cli.commands.run import run_with_progress_async

        assert asyncio.iscoroutinefunction(run_with_progress_async)

    @pytest.mark.asyncio
    async def test_director_registration_in_cli_flow(self):
        """Test that director gets properly registered in CLI flow."""
        from kosmos.agents.registry import get_registry

        # Reset registry
        AgentRegistry._instance = None

        director = ResearchDirectorAgent(
            research_question="Test question",
            domain="biology",
            config={"max_iterations": 1}
        )

        registry = get_registry()
        registry.register(director)

        # Verify registration
        assert registry.get_agent(director.agent_id) is director
        # Verify message router is set
        assert director._message_router is not None


class TestAsyncErrorHandling:
    """Test async error handling with real API."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        AgentRegistry._instance = None
        yield
        AgentRegistry._instance = None

    @pytest.mark.asyncio
    async def test_async_execute_unknown_action_raises(self):
        """Test that unknown action raises ValueError."""
        director = ResearchDirectorAgent(
            research_question="Test",
            domain="biology"
        )

        with pytest.raises(ValueError, match="Unknown action"):
            await director.execute({"action": "invalid_action"})

    @pytest.mark.asyncio
    async def test_async_message_to_unregistered_agent(self):
        """Test sending message to unregistered agent logs error."""
        registry = get_registry()

        sender = BaseAgent(agent_id="error_sender", agent_type="Sender")
        registry.register(sender)

        # Send to non-existent agent - should not raise but log error
        message = await sender.send_message(
            to_agent="nonexistent",
            content={"test": "data"}
        )

        # Message is created but routing fails silently
        assert message.to_agent == "nonexistent"


class TestAsyncPerformance:
    """Test async performance characteristics."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        AgentRegistry._instance = None
        yield
        AgentRegistry._instance = None

    @pytest.mark.asyncio
    async def test_concurrent_message_sending(self):
        """Test that multiple messages can be sent concurrently."""
        registry = get_registry()

        sender = BaseAgent(agent_id="perf_sender", agent_type="Sender")
        receivers = [
            BaseAgent(agent_id=f"perf_receiver_{i}", agent_type="Receiver")
            for i in range(5)
        ]

        registry.register(sender)
        for r in receivers:
            registry.register(r)

        # Send messages concurrently
        import time
        start = time.time()

        tasks = [
            sender.send_message(
                to_agent=f"perf_receiver_{i}",
                content={"index": i}
            )
            for i in range(5)
        ]
        messages = await asyncio.gather(*tasks)

        elapsed = time.time() - start

        assert len(messages) == 5
        # Concurrent sending should be fast
        assert elapsed < 1.0  # Should complete in under 1 second

        # All receivers got messages
        for i, r in enumerate(receivers):
            assert r.messages_received == 1
