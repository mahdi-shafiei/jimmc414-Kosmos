# Getting Started with Kosmos

This guide helps you quickly set up and run Kosmos, the autonomous AI scientist system.

## Prerequisites

- Python 3.11+
- pip package manager
- An Anthropic API key (for production use)

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/jimmc414/Kosmos.git
cd Kosmos

# Initialize submodule for scientific skills
git submodule update --init --recursive

# Install dependencies
pip install -e .
pip install pytest pytest-asyncio  # For testing
```

## Running Your First Research Workflow

### 1. Basic Usage (Mock Mode)

Run without API keys for testing:

```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def run_research():
    workflow = ResearchWorkflow(
        research_objective="Investigate KRAS mutations in cancer drug resistance",
        artifacts_dir="./artifacts"
    )

    result = await workflow.run(num_cycles=2, tasks_per_cycle=5)

    print(f"Completed {result['cycles_completed']} cycles")
    print(f"Generated {result['total_findings']} findings")
    print(f"Validated {result['validated_findings']} findings")

asyncio.run(run_research())
```

### 2. Production Usage (With API Keys)

```python
import asyncio
from anthropic import Anthropic
from kosmos.workflow.research_loop import ResearchWorkflow

async def run_production_research():
    # Initialize Anthropic client
    client = Anthropic()  # Uses ANTHROPIC_API_KEY env variable

    workflow = ResearchWorkflow(
        research_objective="Investigate KRAS mutations in cancer drug resistance",
        anthropic_client=client,
        artifacts_dir="./artifacts"
    )

    result = await workflow.run(num_cycles=5, tasks_per_cycle=10)

    # Generate research report
    report = await workflow.generate_report()
    with open("research_report.md", "w") as f:
        f.write(report)

asyncio.run(run_production_research())
```

## Key Components

### Gap Implementation Modules

Kosmos implements 6 foundational gaps from the original paper:

| Gap | Component | Description |
|-----|-----------|-------------|
| Gap 0 | `kosmos/compression/` | Context compression (20:1 ratio) |
| Gap 1 | `kosmos/world_model/artifacts.py` | State management with JSON artifacts |
| Gap 2 | `kosmos/orchestration/` | Plan creation, review, novelty detection |
| Gap 3 | `kosmos/agents/skill_loader.py` | Domain-specific scientific skills |
| Gap 4 | Mock implementations | Sandboxed execution (Phase 2) |
| Gap 5 | `kosmos/validation/` | ScholarEval 8-dimension validation |

### Using Individual Components

**Context Compression:**
```python
from kosmos.compression import ContextCompressor

compressor = ContextCompressor()
result = compressor.compress_notebook("/path/to/notebook.ipynb")
print(f"Summary: {result.summary}")
print(f"Statistics: {result.statistics}")
```

**Skill Loading:**
```python
from kosmos.agents import SkillLoader

loader = SkillLoader()
skills = loader.load_skills_for_task("single_cell_analysis")
print(f"Loaded {len(skills)} skill prompts")
```

**ScholarEval Validation:**
```python
from kosmos.validation import ScholarEvalValidator

validator = ScholarEvalValidator()
finding = {
    "summary": "KRAS G12D mutation correlates with poor prognosis",
    "statistics": {"p_value": 0.001, "sample_size": 500}
}
score = await validator.evaluate_finding(finding)
print(f"Overall score: {score.overall_score}")
```

## Running Tests

```bash
# Run all gap module tests (339 tests)
python -m pytest tests/unit/compression/ tests/unit/orchestration/ \
    tests/unit/validation/ tests/unit/workflow/ \
    tests/unit/world_model/test_artifacts.py \
    tests/unit/agents/test_skill_loader.py \
    tests/integration/ tests/e2e/ -v

# Run quick smoke tests
python -m pytest tests/unit/orchestration/ tests/unit/workflow/ -v --tb=short
```

## Configuration

Environment variables:
- `ANTHROPIC_API_KEY`: API key for Anthropic Claude
- `OPENAI_API_KEY`: API key for OpenAI (optional)

Configuration file: `config.yaml` (optional)
```yaml
provider: anthropic
model: claude-3-5-sonnet-20241022
max_cycles: 20
artifacts_dir: ./artifacts
```

## Documentation

- [README.md](README.md) - Project overview
- [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) - Architecture details
- [TESTS_STATUS.md](TESTS_STATUS.md) - Test suite status
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) - Gap analysis

## Support

- Issues: https://github.com/jimmc414/Kosmos/issues
- Documentation: See `/docs` directory
