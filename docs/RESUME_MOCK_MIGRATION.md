# Resume: Mock to Real Test Migration

## Quick Start
```
@docs/RESUME_MOCK_MIGRATION.md fix Phase 3 bugs
```

## Context
Converting mock-based tests to real API/service calls for production readiness.

## Completed
| Phase | Tests | Status |
|-------|-------|--------|
| Phase 1: Core LLM | 43 | ✓ Complete |
| Phase 2: Knowledge Layer | 57 | ✓ Complete |
| Phase 3: Agent Tests | 124 (4 skipped) | Tests pass, bugs need fixing |
| **Total** | **224** | |

## Current Task: Fix Phase 3 Bugs

### Bug 1: `generate_structured` interface mismatch

**Location:** `kosmos/agents/literature_analyzer.py:265-270`

**Problem:** Agent passes `max_tokens=2048` to `generate_structured()` but `ClaudeClient.generate_structured()` doesn't accept this parameter.

```python
# Current (broken):
analysis = self.llm_client.generate_structured(
    prompt=prompt,
    output_schema=self._get_summarization_schema(),
    system="...",
    max_tokens=2048  # <-- ClaudeClient doesn't accept this
)
```

**Fix options:**
1. Remove `max_tokens` from the call (ClaudeClient uses default)
2. Add `max_tokens` parameter to `ClaudeClient.generate_structured()`

### Bug 2: Provider parameter name mismatch

**Location:**
- `kosmos/core/llm.py:403-408` - ClaudeClient uses `output_schema`
- `kosmos/core/providers/openai.py:449-456` - LiteLLMProvider uses `schema`

**Problem:** Different providers use different parameter names for the same thing.

**Fix:** Standardize on one name across all providers (recommend `output_schema` to match existing agent code).

### After Fixing
Remove the `@pytest.mark.skip` decorators from these tests:
- `tests/unit/agents/test_literature_analyzer.py:87` - `test_summarize_paper`
- `tests/unit/agents/test_literature_analyzer.py:102` - `test_summarize_paper_with_minimal_abstract`
- `tests/unit/agents/test_literature_analyzer.py:176` - `test_agent_execute_summarize`
- `tests/unit/agents/test_literature_analyzer.py:196` - `test_real_paper_summarization`

### Verification
```bash
# After fixing, all 10 tests should pass (not 6 passed + 4 skipped)
pytest tests/unit/agents/test_literature_analyzer.py -v --no-cov
```

## After Bug Fixes: Phase 4 - Integration Tests
1. `tests/integration/test_analysis_pipeline.py`
2. `tests/integration/test_phase2_e2e.py`
3. `tests/integration/test_phase3_e2e.py`
4. `tests/integration/test_concurrent_research.py`

## Reference
- Full checkpoint: `docs/CHECKPOINT_MOCK_MIGRATION.md`
