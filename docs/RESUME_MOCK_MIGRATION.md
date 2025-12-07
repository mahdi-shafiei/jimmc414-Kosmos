# Resume Mock to Real Test Migration

## Context
Converting mock-based tests to real LLM API calls for production readiness.

## Completed
- **Phase 1: Core LLM tests** - 43 tests pass with real APIs
- **Phase 2: Knowledge Layer tests** - 57 tests pass with real services

## Current Status
Ready to continue with Phase 3: Agent tests.
- ANTHROPIC_API_KEY: Configured
- SEMANTIC_SCHOLAR_API_KEY: Configured (rate limit: 1 req/sec)
- Neo4j: Running (kosmos-neo4j)
- ChromaDB: Available

## Resume Task: Phase 3 - Agents

### Files to Convert
1. `tests/unit/agents/test_research_director.py` - Claude API
2. `tests/unit/agents/test_literature_analyzer.py` - Claude API + Neo4j
3. `tests/unit/agents/test_data_analyst.py` - Claude API
4. `tests/unit/agents/test_hypothesis_generator.py` - Claude + Semantic Scholar

### Then Phase 4 - Integration
1. `tests/integration/test_analysis_pipeline.py`
2. `tests/integration/test_phase2_e2e.py`
3. `tests/integration/test_phase3_e2e.py`
4. `tests/integration/test_concurrent_research.py`

### Pattern
```python
import os, pytest, uuid

pytestmark = [
    pytest.mark.requires_claude,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires API key")
]

def unique_prompt(base: str) -> str:
    return f"{base} [test-id: {uuid.uuid4().hex[:8]}]"
```

### Fixtures Available
- `real_anthropic_client`, `deepseek_client` - LLM clients
- `real_vector_db` - Ephemeral ChromaDB
- `real_embedder` - SentenceTransformer
- `real_knowledge_graph` - Neo4j

### Verify
```bash
pytest tests/unit/knowledge/ -v --no-cov
```

## Note on Semantic Scholar
Works without API key (just lower rate limits). When key arrives, add to .env for higher limits.
