# Kosmos Runbook Generation Prompt

## Purpose

This document is a comprehensive prompt for generating an operational runbook for Kosmos, an autonomous AI scientist for scientific discovery. Use this with Claude or another capable LLM to generate a complete, production-ready runbook.

---

## Prompt

You are tasked with generating a comprehensive operational runbook for Kosmos v0.2.0, an autonomous AI scientist system. The runbook should be suitable for operators deploying and running Kosmos in production environments.

### System Overview

Kosmos is an open-source implementation of an autonomous AI scientist that:
- Generates testable hypotheses from literature and data analysis
- Designs experimental protocols to test hypotheses
- Executes code in Docker-sandboxed containers (Python and R supported)
- Validates discoveries using an 8-dimension ScholarEval quality framework
- Builds knowledge graphs in Neo4j to track relationships between concepts
- Streams real-time events via SSE/WebSocket for monitoring

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DISCOVERY ORCHESTRATOR                             │
│                   (ResearchDirectorAgent)                               │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  Hypothesis │  │  Experiment │  │    Data     │  │  Literature │   │
│  │  Generator  │  │  Designer   │  │   Analyst   │  │  Analyzer   │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         └────────────────┴────────────────┴────────────────┘           │
│                                   │                                     │
│                    ┌──────────────▼──────────────┐                     │
│                    │   EVENT BUS (Streaming)     │                     │
│                    │  SSE | WebSocket | CLI      │                     │
│                    └──────────────┬──────────────┘                     │
│                                   │                                     │
│                    ┌──────────────▼──────────────┐                     │
│                    │    STATE MANAGER            │                     │
│                    │  JSON | Neo4j | Vector DB   │                     │
│                    └──────────────┬──────────────┘                     │
│                                   │                                     │
│                    ┌──────────────▼──────────────┐                     │
│                    │     EXECUTION SANDBOX       │                     │
│                    │  Docker | Python | R        │                     │
│                    └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Files and Their Purposes

| Component | File | Description |
|-----------|------|-------------|
| Research Director | `kosmos/agents/research_director.py` | Master orchestrator coordinating all agents |
| Hypothesis Generator | `kosmos/agents/hypothesis_generator.py` | Generates testable hypotheses |
| Experiment Designer | `kosmos/agents/experiment_designer.py` | Creates experimental protocols |
| Data Analyst | `kosmos/agents/data_analyst.py` | Interprets experimental results |
| Literature Analyzer | `kosmos/agents/literature_analyzer.py` | Searches and synthesizes papers |
| Research Workflow | `kosmos/workflow/research_loop.py` | Main async research loop |
| Plan Creator | `kosmos/orchestration/plan_creator.py` | Generates strategic tasks |
| Plan Reviewer | `kosmos/orchestration/plan_reviewer.py` | Validates plan quality |
| State Manager | `kosmos/world_model/artifacts.py` | 4-layer hybrid state storage |
| ScholarEval | `kosmos/validation/scholar_eval.py` | 8-dimension discovery validation |
| Context Compressor | `kosmos/compression/compressor.py` | 20:1 context compression |
| Skill Loader | `kosmos/agents/skill_loader.py` | 116 domain-specific skills |
| Event Bus | `kosmos/core/event_bus.py` | Real-time event pub/sub |
| Events | `kosmos/core/events.py` | 18 event types for streaming |
| SSE Endpoint | `kosmos/api/streaming.py` | Server-Sent Events streaming |
| WebSocket Endpoint | `kosmos/api/websocket.py` | Bidirectional event streaming |
| CLI Streaming | `kosmos/cli/streaming.py` | Rich-based progress display |
| Docker Sandbox | `kosmos/execution/sandbox.py` | Secure code execution |
| R Executor | `kosmos/execution/r_executor.py` | R language execution support |
| LLM Providers | `kosmos/core/providers/` | Anthropic, OpenAI, LiteLLM |

### Infrastructure Requirements

**Required:**
- Python 3.11+
- Docker (for sandboxed code execution)
- One LLM API key (Anthropic, OpenAI, or LiteLLM-compatible)

**Optional but Recommended:**
- Neo4j 5.x (knowledge graph)
- Redis 7.x (distributed caching)
- PostgreSQL 15+ (production database, SQLite default)

### CLI Commands

```bash
# Run research
kosmos run "Research question" --domain biology --max-iterations 10

# With real-time streaming display
kosmos run "Research question" --stream

# Streaming without token display
kosmos run "Research question" --stream --no-stream-tokens

# Interactive mode
kosmos run --interactive

# With budget limit
kosmos run "Research question" --budget 50

# Save results
kosmos run "Research question" --output results.json

# System diagnostics
kosmos doctor

# System info
kosmos info
```

### API Endpoints

```bash
# SSE streaming
GET /stream/events?process_id=<id>&types=workflow.progress,llm.token

# WebSocket streaming
WS /ws/events?process_id=<id>

# Health check
GET /stream/health
GET /ws/connections
```

### Python API

```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow
from kosmos.core.event_bus import get_event_bus
from kosmos.core.events import EventType

# Subscribe to events
event_bus = get_event_bus()
event_bus.subscribe(
    callback=my_handler,
    event_types=[EventType.WORKFLOW_PROGRESS, EventType.LLM_TOKEN],
    process_ids=["research_abc"]
)

# Run research
async def run():
    workflow = ResearchWorkflow(
        research_objective="Your research question",
        artifacts_dir="./artifacts"
    )
    result = await workflow.run(num_cycles=5, tasks_per_cycle=10)
    report = await workflow.generate_report()
    return report

asyncio.run(run())
```

### Environment Variables

```bash
# LLM Provider (required - one of these)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
# OR
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# OR
LLM_PROVIDER=litellm
LITELLM_MODEL=ollama/llama3.1:8b

# Database
DATABASE_URL=sqlite:///kosmos.db
# OR for production:
DATABASE_URL=postgresql://kosmos:password@localhost:5432/kosmos

# Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=kosmos-password

# Redis (optional)
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379/0

# Budget
BUDGET_ENABLED=true
BUDGET_LIMIT_USD=10.00

# Research settings
MAX_RESEARCH_ITERATIONS=10
ENABLED_DOMAINS=biology,physics,chemistry,neuroscience,materials

# Concurrency
ENABLE_CONCURRENT_OPERATIONS=true
MAX_CONCURRENT_EXPERIMENTS=10
MAX_PARALLEL_HYPOTHESES=3
MAX_CONCURRENT_LLM_CALLS=5

# Sandbox
ENABLE_SANDBOXING=true
MAX_EXPERIMENT_EXECUTION_TIME=300

# Debug
DEBUG_MODE=false
DEBUG_LEVEL=0
```

### Docker Services

```bash
# Start all services
docker compose --profile dev up -d

# Individual services
docker compose up -d neo4j    # Knowledge graph
docker compose up -d redis    # Caching
docker compose up -d postgres # Production database

# Service URLs
# Neo4j Browser: http://localhost:7474 (neo4j/kosmos-password)
# PostgreSQL: localhost:5432 (kosmos/kosmos-dev-password)
# Redis: localhost:6379
```

### Workflow States

```
INITIALIZING → GENERATING_HYPOTHESES → DESIGNING_EXPERIMENTS → EXECUTING
                                                                    │
                                                                    ▼
CONVERGED ← REFINING ← ANALYZING ←─────────────────────────────────┘
```

### Event Types for Streaming

| Event Type | Description |
|------------|-------------|
| `workflow.started` | Research workflow began |
| `workflow.progress` | Progress update (cycle, percentage) |
| `workflow.completed` | Workflow finished |
| `cycle.started` | Research cycle began |
| `cycle.completed` | Cycle finished with findings |
| `task.started` | Individual task began |
| `task.completed` | Task finished |
| `llm.call_started` | LLM API call initiated |
| `llm.token` | Streaming token received |
| `llm.call_completed` | LLM call finished |
| `execution.executing` | Code executing in sandbox |
| `execution.output` | Execution output line |
| `execution.completed` | Execution finished |
| `stage.started` | Processing stage began |
| `stage.completed` | Stage finished |

### Test Coverage

- 3704+ tests total
- Unit tests: 2251
- Integration tests: 415
- E2E tests: 121
- Requirements tests: 815

### Runbook Sections Required

Generate a comprehensive runbook with the following sections:

1. **Quick Start Guide**
   - Installation steps
   - Environment setup
   - First research run
   - Verification steps

2. **System Architecture**
   - Component diagram
   - Data flow
   - State management
   - Event streaming architecture

3. **Installation and Configuration**
   - Prerequisites
   - Package installation
   - Environment variables (all options)
   - Docker services setup
   - Database initialization

4. **Operations**
   - Starting the system
   - Running research workflows
   - Monitoring with streaming
   - Stopping gracefully
   - Log locations and formats

5. **CLI Reference**
   - All commands with examples
   - Options and flags
   - Output formats

6. **API Reference**
   - REST endpoints
   - SSE streaming
   - WebSocket protocol
   - Python API examples

7. **Research Workflow Lifecycle**
   - Detailed state machine
   - Each phase explained
   - Convergence criteria
   - Exploration/exploitation balance

8. **Agent Operations**
   - Each agent's role
   - Inputs/outputs
   - Configuration
   - Success criteria

9. **Monitoring and Observability**
   - Real-time streaming setup
   - Event subscription
   - Progress tracking
   - Metrics collection

10. **Code Execution Sandbox**
    - Docker configuration
    - Security model
    - Python execution
    - R language support
    - Resource limits

11. **Knowledge Graph Operations** (Neo4j)
    - Schema
    - Queries
    - Maintenance

12. **Budget and Cost Management**
    - Cost tracking
    - Budget enforcement
    - LLM cost optimization

13. **Troubleshooting Guide**
    - Common issues and solutions
    - Diagnostic commands
    - Log analysis
    - Recovery procedures

14. **Backup and Recovery**
    - What to backup
    - Database backups
    - Artifact preservation
    - Disaster recovery

15. **Security Considerations**
    - API key management
    - Sandbox isolation
    - Network security
    - Data protection

16. **Performance Tuning**
    - Concurrency settings
    - Caching configuration
    - Resource optimization
    - Scaling considerations

17. **Appendices**
    - Complete environment variable reference
    - Event type reference
    - Error code reference
    - Glossary

### Output Format

Generate the runbook in Markdown format with:
- Clear section headers
- Code blocks for commands and configuration
- Tables for structured information
- Mermaid diagrams for architecture
- Practical examples throughout
- Cross-references between sections

### Tone

Write for experienced operators who will deploy and manage Kosmos in production. Be precise, thorough, and practical. Include specific commands, exact file paths, and real configuration values.

---

## Usage

Copy this entire document and provide it to Claude or another capable LLM with the instruction:

> Generate a complete operational runbook for Kosmos based on the specifications in this document. The runbook should be production-ready and comprehensive.

The LLM should generate a 100+ page runbook covering all aspects of operating Kosmos.
