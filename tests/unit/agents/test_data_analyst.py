"""
Tests for Data Analyst Agent.

Tests using REAL Claude API (not mocks).
Only interpret_results() uses Claude - other methods are pure Python.
Requires ANTHROPIC_API_KEY environment variable.
"""

import os
import pytest
import json
import uuid
import sys
import platform as plat
from datetime import datetime

from kosmos.agents.data_analyst import DataAnalystAgent, ResultInterpretation
from kosmos.models.result import (
    ExperimentResult,
    ResultStatus,
    StatisticalTestResult,
    VariableResult,
    ExecutionMetadata
)
from kosmos.models.hypothesis import Hypothesis


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


def make_metadata(experiment_id: str, protocol_id: str) -> ExecutionMetadata:
    """Create ExecutionMetadata with all required fields."""
    return ExecutionMetadata(
        experiment_id=experiment_id,
        protocol_id=protocol_id,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        duration_seconds=1.0,
        python_version=sys.version.split()[0],
        platform=plat.system(),
        random_seed=42
    )


def make_result(
    p_value: float,
    effect_size: float,
    supports_hypothesis: bool = None,
    variable_results: list = None
) -> ExperimentResult:
    """Helper to create experiment results with specific values."""
    exp_id = f"exp-{unique_id()}"
    proto_id = f"proto-{unique_id()}"
    if supports_hypothesis is None:
        supports_hypothesis = p_value < 0.05
    return ExperimentResult(
        id=f"result-{unique_id()}",
        experiment_id=exp_id,
        hypothesis_id=f"hyp-{unique_id()}",
        protocol_id=proto_id,
        status=ResultStatus.SUCCESS,
        primary_test="T-test",
        primary_p_value=p_value,
        primary_effect_size=effect_size,
        supports_hypothesis=supports_hypothesis,
        statistical_tests=[
            StatisticalTestResult(
                test_type="t-test",
                test_name="T-test",
                statistic=2.0,
                p_value=p_value,
                effect_size=effect_size,
                significant_0_05=p_value < 0.05,
                significant_0_01=p_value < 0.01,
                significant_0_001=p_value < 0.001,
                significance_label="*" if p_value < 0.05 else "ns",
                is_primary=True
            )
        ],
        variable_results=variable_results or [],
        metadata=make_metadata(exp_id, proto_id),
        created_at=datetime.utcnow()
    )


# Fixtures

@pytest.fixture
def data_analyst_agent():
    """Create DataAnalystAgent with real Claude client."""
    agent = DataAnalystAgent(config={
        "model": "claude-3-haiku-20240307",
        "use_literature_context": True,
        "detailed_interpretation": True,
        "anomaly_detection_enabled": True,
        "pattern_detection_enabled": True
    })
    return agent


@pytest.fixture
def sample_experiment_result():
    """Create sample successful experiment result."""
    exp_id = f"exp-{unique_id()}"
    proto_id = f"proto-{unique_id()}"
    return ExperimentResult(
        id=f"result-{unique_id()}",
        experiment_id=exp_id,
        hypothesis_id=f"hyp-{unique_id()}",
        protocol_id=proto_id,
        status=ResultStatus.SUCCESS,
        primary_test="Two-sample T-test",
        primary_p_value=0.012,
        primary_effect_size=0.65,
        primary_ci_lower=0.2,
        primary_ci_upper=1.1,
        supports_hypothesis=True,
        statistical_tests=[
            StatisticalTestResult(
                test_type="t-test",
                test_name="Two-sample T-test",
                statistic=2.54,
                p_value=0.012,
                effect_size=0.65,
                effect_size_type="Cohen's d",
                confidence_interval={"lower": 0.2, "upper": 1.1},
                sample_size=100,
                degrees_of_freedom=98,
                significant_0_05=True,
                significant_0_01=False,
                significant_0_001=False,
                significance_label="*",
                is_primary=True
            )
        ],
        variable_results=[
            VariableResult(
                variable_name="treatment",
                variable_type="independent",
                mean=10.5,
                median=10.3,
                std=2.1,
                min=6.2,
                max=15.8,
                q1=9.1,
                q3=11.9,
                n_samples=50,
                n_missing=0
            ),
            VariableResult(
                variable_name="control",
                variable_type="independent",
                mean=8.8,
                median=8.5,
                std=2.3,
                min=4.5,
                max=13.2,
                q1=7.2,
                q3=10.1,
                n_samples=50,
                n_missing=0
            )
        ],
        metadata=make_metadata(exp_id, proto_id),
        raw_data={"mean_diff": 1.7},
        generated_files=[],
        version=1,
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_hypothesis():
    """Create sample hypothesis."""
    return Hypothesis(
        id=f"hyp-{unique_id()}",
        research_question="Does treatment X increase outcome Y compared to control?",
        statement="Treatment X increases outcome Y compared to control",
        rationale="Prior studies suggest mechanism via pathway Z which would mediate this effect",
        domain="biology",
        testability_score=0.9,
        novelty_score=0.7
    )


# Result Interpretation Tests (uses Claude API)

@pytest.mark.unit
class TestResultInterpretation:
    """Tests for result interpretation functionality using real Claude."""

    def test_interpret_results_success(self, data_analyst_agent, sample_experiment_result,
                                       sample_hypothesis):
        """Test successful result interpretation with real Claude."""
        interpretation = data_analyst_agent.interpret_results(
            result=sample_experiment_result,
            hypothesis=sample_hypothesis,
            literature_context="Prior work shows similar effects in related domains."
        )

        # Verify structure (values vary with real API)
        assert isinstance(interpretation, ResultInterpretation)
        assert interpretation.experiment_id == sample_experiment_result.experiment_id
        assert isinstance(interpretation.hypothesis_supported, bool)
        assert 0 <= interpretation.confidence <= 1
        assert isinstance(interpretation.summary, str)
        assert len(interpretation.summary) > 0
        assert isinstance(interpretation.key_findings, list)

    def test_interpret_results_without_hypothesis(self, data_analyst_agent, sample_experiment_result):
        """Test interpretation without hypothesis."""
        interpretation = data_analyst_agent.interpret_results(
            result=sample_experiment_result,
            hypothesis=None,
            literature_context=None
        )

        assert isinstance(interpretation, ResultInterpretation)
        assert interpretation.experiment_id == sample_experiment_result.experiment_id
        assert isinstance(interpretation.summary, str)

    def test_interpret_results_returns_follow_ups(self, data_analyst_agent, sample_experiment_result,
                                                  sample_hypothesis):
        """Test that interpretation includes follow-up suggestions."""
        interpretation = data_analyst_agent.interpret_results(
            result=sample_experiment_result,
            hypothesis=sample_hypothesis
        )

        assert isinstance(interpretation, ResultInterpretation)
        # Real Claude should provide some follow-up suggestions
        assert isinstance(interpretation.follow_up_experiments, list)

    def test_extract_result_summary(self, data_analyst_agent, sample_experiment_result):
        """Test extraction of result summary (pure Python, no API call)."""
        summary = data_analyst_agent._extract_result_summary(sample_experiment_result)

        assert summary["experiment_id"] == sample_experiment_result.experiment_id
        assert summary["status"] == "success"
        assert summary["primary_test"] == "Two-sample T-test"
        assert summary["primary_p_value"] == 0.012
        assert summary["primary_effect_size"] == 0.65
        assert len(summary["statistical_tests"]) == 1
        assert len(summary["variables"]) == 2

    def test_build_interpretation_prompt(self, data_analyst_agent, sample_experiment_result,
                                        sample_hypothesis):
        """Test interpretation prompt building (pure Python, no API call)."""
        summary = data_analyst_agent._extract_result_summary(sample_experiment_result)
        prompt = data_analyst_agent._build_interpretation_prompt(
            summary, sample_hypothesis, "Literature context..."
        )

        assert "HYPOTHESIS:" in prompt
        assert sample_hypothesis.statement in prompt
        assert "EXPERIMENTAL RESULTS:" in prompt
        assert "P-value: 0.012" in prompt
        assert "LITERATURE CONTEXT:" in prompt
        assert "Format your response as JSON" in prompt


# Anomaly Detection Tests (pure Python, no API calls)

@pytest.mark.unit
class TestAnomalyDetection:
    """Tests for anomaly detection functionality (no Claude needed)."""

    def test_detect_anomaly_tiny_effect_significant_p(self, data_analyst_agent):
        """Test detection of significant p-value with tiny effect size."""
        result = make_result(p_value=0.001, effect_size=0.05)  # < 0.01 and < 0.3

        anomalies = data_analyst_agent.detect_anomalies(result)

        assert len(anomalies) > 0
        # The actual message contains "tiny effect size"
        assert any("tiny effect size" in a.lower() for a in anomalies)

    def test_detect_anomaly_large_effect_nonsignificant_p(self, data_analyst_agent):
        """Test detection of large effect size with non-significant p-value."""
        result = make_result(p_value=0.15, effect_size=0.8)  # > 0.05 and > 0.5

        anomalies = data_analyst_agent.detect_anomalies(result)

        assert len(anomalies) > 0
        # The actual message contains "Large effect size"
        assert any("large effect size" in a.lower() for a in anomalies)

    def test_detect_anomaly_pvalue_zero(self, data_analyst_agent):
        """Test detection of p-value exactly 0."""
        result = make_result(p_value=0.0, effect_size=2.5)

        anomalies = data_analyst_agent.detect_anomalies(result)

        assert len(anomalies) > 0
        # The actual message says "is exactly 0.0"
        assert any("exactly 0.0" in a for a in anomalies)

    def test_detect_anomaly_high_variability(self, data_analyst_agent):
        """Test detection of high variability in variables."""
        variable_results = [
            VariableResult(
                variable_name="highly_variable",
                variable_type="dependent",
                mean=10.0,
                median=9.5,
                std=15.0,  # Std > mean (CV > 1.0)
                min=0.1,
                max=50.0,
                n_samples=30,
                n_missing=0
            )
        ]
        result = make_result(p_value=0.05, effect_size=0.5, variable_results=variable_results)

        anomalies = data_analyst_agent.detect_anomalies(result)

        assert len(anomalies) > 0
        # The actual message says "has very high variability"
        assert any("high variability" in a.lower() for a in anomalies)

    def test_detect_no_anomalies(self, data_analyst_agent, sample_experiment_result):
        """Test normal result with no anomalies."""
        anomalies = data_analyst_agent.detect_anomalies(sample_experiment_result)

        # Should have no major anomalies (or only minor notes)
        assert len(anomalies) == 0 or all("NOTE:" in a for a in anomalies)


# Pattern Detection Tests (pure Python, no API calls)

@pytest.mark.unit
class TestPatternDetection:
    """Tests for pattern detection across multiple results (no Claude needed)."""

    def test_detect_pattern_consistent_positive_effects(self, data_analyst_agent):
        """Test detection of consistent positive effects (requires >= 3 results)."""
        results = [make_result(0.01, 0.5 + i * 0.1) for i in range(5)]

        patterns = data_analyst_agent.detect_patterns_across_results(results)

        assert len(patterns) > 0
        # The message says "positive effects"
        assert any("positive effects" in p.lower() for p in patterns)

    def test_detect_pattern_increasing_trend(self, data_analyst_agent):
        """Test detection of increasing effect size trend (requires >= 4 results)."""
        # Create monotonically increasing effect sizes
        results = [make_result(0.01, 0.2 * (i + 1)) for i in range(4)]

        patterns = data_analyst_agent.detect_patterns_across_results(results)

        assert len(patterns) > 0
        # The message says "increasing trend"
        assert any("increasing trend" in p.lower() for p in patterns)

    def test_detect_pattern_bimodal_pvalues(self, data_analyst_agent):
        """Test detection of bimodal p-value distribution (requires >= 5 results)."""
        # Create bimodal distribution: some very significant, some clearly non-significant
        p_values = [0.001, 0.005, 0.002, 0.15, 0.20, 0.18]  # 3 < 0.01, 3 > 0.1
        results = [make_result(p, 0.5) for p in p_values]

        patterns = data_analyst_agent.detect_patterns_across_results(results)

        assert len(patterns) > 0
        # The message says "Bimodal p-value distribution"
        assert any("bimodal" in p.lower() for p in patterns)

    def test_detect_patterns_insufficient_data(self, data_analyst_agent):
        """Test pattern detection with insufficient data (< 2 results)."""
        results = [make_result(0.01, 0.5)]

        patterns = data_analyst_agent.detect_patterns_across_results(results)

        # Should return empty list for single result
        assert len(patterns) == 0


# Significance Interpretation Tests (pure Python, no API calls)

@pytest.mark.unit
class TestSignificanceInterpretation:
    """Tests for statistical significance interpretation (no Claude needed)."""

    def test_interpret_very_significant(self, data_analyst_agent):
        """Test interpretation of very significant result."""
        interpretation = data_analyst_agent.interpret_significance(
            p_value=0.0001,
            effect_size=0.8,
            sample_size=100
        )

        assert "very strong evidence" in interpretation.lower()
        assert "p < 0.001" in interpretation.lower()
        assert "large" in interpretation.lower()

    def test_interpret_significant_small_effect(self, data_analyst_agent):
        """Test interpretation of significant but small effect."""
        interpretation = data_analyst_agent.interpret_significance(
            p_value=0.01,
            effect_size=0.15,
            sample_size=1000
        )

        # Check for evidence against null (moderate or strong)
        assert "evidence" in interpretation.lower()
        # Check for either "negligible" or "small" effect size description
        assert "negligible" in interpretation.lower() or "small" in interpretation.lower()
        # Check for large sample warning
        assert "large sample" in interpretation.lower() or "sample size" in interpretation.lower()

    def test_interpret_nonsignificant_large_effect(self, data_analyst_agent):
        """Test interpretation of non-significant but large effect."""
        interpretation = data_analyst_agent.interpret_significance(
            p_value=0.08,
            effect_size=0.75,
            sample_size=20
        )

        # Check for interpretation of borderline significance
        assert "suggestive" in interpretation.lower() or "not" in interpretation.lower()
        assert "large" in interpretation.lower() or "medium" in interpretation.lower()
        assert "small sample" in interpretation.lower()

    def test_interpret_small_sample_warning(self, data_analyst_agent):
        """Test warning for small sample size."""
        interpretation = data_analyst_agent.interpret_significance(
            p_value=0.05,
            effect_size=0.5,
            sample_size=15
        )

        assert "small sample" in interpretation.lower()
        assert "caution" in interpretation.lower()


# Agent Lifecycle Tests

@pytest.mark.unit
class TestAgentLifecycle:
    """Tests for agent lifecycle and task execution."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = DataAnalystAgent(config={
            "model": "claude-3-haiku-20240307",
            "use_literature_context": False,
            "detailed_interpretation": True
        })

        assert agent.agent_type == "DataAnalystAgent"
        assert agent.use_literature_context is False
        assert agent.detailed_interpretation is True
        assert agent.interpretation_history == []

    def test_execute_interpret_results_task(self, data_analyst_agent, sample_experiment_result):
        """Test execute() with interpret_results action using real Claude."""
        task = {
            "action": "interpret_results",
            "result": sample_experiment_result,
            "hypothesis": None,
            "literature_context": None
        }

        result = data_analyst_agent.execute(task)

        assert result["success"] is True
        assert "interpretation" in result
        assert result["interpretation"]["experiment_id"] == sample_experiment_result.experiment_id

    def test_execute_detect_anomalies_task(self, data_analyst_agent, sample_experiment_result):
        """Test execute() with detect_anomalies action (no Claude needed)."""
        task = {
            "action": "detect_anomalies",
            "result": sample_experiment_result
        }

        result = data_analyst_agent.execute(task)

        assert result["success"] is True
        assert "anomalies" in result
        assert isinstance(result["anomalies"], list)

    def test_execute_detect_patterns_task(self, data_analyst_agent):
        """Test execute() with detect_patterns action (no Claude needed)."""
        results = [make_result(0.01, 0.5 + i * 0.1) for i in range(3)]

        task = {
            "action": "detect_patterns",
            "results": results
        }

        result = data_analyst_agent.execute(task)

        assert result["success"] is True
        assert "patterns" in result
        assert isinstance(result["patterns"], list)

    def test_execute_unknown_action(self, data_analyst_agent):
        """Test execute() with unknown action."""
        task = {
            "action": "unknown_action"
        }

        result = data_analyst_agent.execute(task)

        assert result["success"] is False
        assert "error" in result


# ResultInterpretation Class Tests (pure Python, no API calls)

@pytest.mark.unit
class TestResultInterpretationClass:
    """Tests for ResultInterpretation data class."""

    def test_result_interpretation_to_dict(self):
        """Test ResultInterpretation to_dict() method."""
        interpretation = ResultInterpretation(
            experiment_id=f"exp-{unique_id()}",
            hypothesis_supported=True,
            confidence=0.85,
            summary="Test summary",
            key_findings=["Finding 1", "Finding 2"],
            significance_interpretation="Significant",
            biological_significance="Meaningful",
            comparison_to_prior_work="Similar to prior work",
            potential_confounds=["Confound 1"],
            follow_up_experiments=["Experiment 1"],
            anomalies_detected=[],
            patterns_detected=[],
            overall_assessment="Good quality"
        )

        result_dict = interpretation.to_dict()

        assert "exp-" in result_dict["experiment_id"]
        assert result_dict["hypothesis_supported"] is True
        assert result_dict["confidence"] == 0.85
        assert len(result_dict["key_findings"]) == 2
        assert "created_at" in result_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
