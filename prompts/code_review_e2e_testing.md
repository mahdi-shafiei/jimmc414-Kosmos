# Code Review Prompt: Comprehensive E2E Testing Analysis

## Objective

Perform a comprehensive code review of the Kosmos codebase focusing specifically on end-to-end (E2E) testing coverage, quality, and gaps. The goal is to identify what's tested, what's missing, and recommend improvements.

---

## Codebase Context

**Project**: Kosmos - Autonomous AI Scientist
**Key Components**:
- Research Director Agent (orchestrator)
- Hypothesis Generator, Experiment Designer, Data Analyst agents
- World Model (Neo4j-based entity storage)
- LLM Providers (OpenAI, Anthropic)
- Knowledge Graph (Neo4j)
- Code Executor (sandboxed execution)
- Metrics/Budget tracking

**Test Locations**:
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests
- `tests/e2e/` - End-to-end tests
- `tests/conftest.py` - Shared fixtures and markers

---

## Review Tasks

### 1. Current E2E Test Inventory

Analyze `tests/e2e/` and answer:

- [ ] List all E2E test files and their purposes
- [ ] Count total E2E test cases (methods starting with `test_`)
- [ ] Identify which tests are skipped and why (check `@pytest.mark.skip`, `@pytest.mark.skipif`)
- [ ] Map tests to the components they cover
- [ ] Identify tests that require external dependencies (Neo4j, API keys, Docker)

### 2. Critical Path Coverage

Evaluate coverage of the main research workflow:

```
Research Question → Hypotheses → Experiments → Results → Analysis → Refinement → Convergence
```

For each stage, determine:
- [ ] Is there an E2E test that exercises this stage?
- [ ] Does the test use real components or mocks?
- [ ] What failure modes are tested?
- [ ] Are edge cases covered (empty results, errors, timeouts)?

### 3. Integration Point Testing

Review tests for critical integration points:

| Integration Point | Test Exists? | Quality |
|-------------------|--------------|---------|
| Agent ↔ LLM Provider | ? | ? |
| Agent ↔ World Model | ? | ? |
| Agent ↔ Knowledge Graph | ? | ? |
| Research Director ↔ Sub-agents | ? | ? |
| Executor ↔ Sandbox | ? | ? |
| CLI ↔ Core API | ? | ? |
| Metrics ↔ Budget Enforcement | ? | ? |

### 4. Test Quality Assessment

For existing E2E tests, evaluate:

- [ ] **Assertions**: Are assertions specific and meaningful?
- [ ] **Setup/Teardown**: Is test data properly cleaned up?
- [ ] **Isolation**: Do tests affect each other?
- [ ] **Determinism**: Can tests produce flaky results?
- [ ] **Timeouts**: Are long-running operations bounded?
- [ ] **Error Messages**: Do failures provide useful diagnostics?

### 5. Missing Test Scenarios

Identify gaps in coverage:

**Happy Path Gaps**:
- [ ] Full research cycle with real LLM (if API key available)
- [ ] Multi-iteration convergence
- [ ] Knowledge graph accumulation across cycles
- [ ] Concurrent hypothesis evaluation

**Error Handling Gaps**:
- [ ] LLM provider failures mid-workflow
- [ ] Budget exceeded mid-research
- [ ] Database connection loss
- [ ] Invalid hypothesis/experiment data
- [ ] Sandbox execution failures

**Edge Case Gaps**:
- [ ] Empty research results
- [ ] Single hypothesis workflow
- [ ] Max iterations reached without convergence
- [ ] Recovery from ERROR state

### 6. Test Infrastructure Review

Evaluate the testing infrastructure:

- [ ] **Fixtures** (`conftest.py`): Are they reusable and well-documented?
- [ ] **Markers**: Are custom markers (`@pytest.mark.e2e`, `@pytest.mark.slow`) used consistently?
- [ ] **Mocking Strategy**: Is there a clear pattern for when to mock vs. use real components?
- [ ] **CI/CD Integration**: Can tests run in CI? What environment variables are required?
- [ ] **Test Data**: Is test data managed appropriately (fixtures, factories, builders)?

### 7. Performance and Resource Considerations

- [ ] How long do E2E tests take to run?
- [ ] Which tests require expensive resources (API calls, Docker, databases)?
- [ ] Is there a "smoke test" subset for quick validation?
- [ ] Are tests parallelizable?

---

## Deliverables

### Summary Report

Provide:

1. **Coverage Score** (0-100): Estimated E2E coverage of critical paths
2. **Top 5 Missing Tests**: Highest-priority E2E tests to add
3. **Top 5 Quality Issues**: Most significant problems in existing tests
4. **Recommended Test Strategy**: How to approach improving E2E coverage

### Test Gap Matrix

| Component | Happy Path | Error Handling | Edge Cases | Priority |
|-----------|------------|----------------|------------|----------|
| Research Director | ✅/❌ | ✅/❌ | ✅/❌ | HIGH/MED/LOW |
| Hypothesis Generator | ✅/❌ | ✅/❌ | ✅/❌ | HIGH/MED/LOW |
| ... | ... | ... | ... | ... |

### Recommended New Tests

For each recommended test, provide:

```markdown
### Test: [Name]

**Purpose**: What this test validates
**Components**: Which components are exercised
**Prerequisites**: Required setup (Neo4j, API keys, etc.)
**Priority**: HIGH/MEDIUM/LOW
**Estimated Effort**: Hours to implement

**Test Outline**:
1. Setup: ...
2. Execute: ...
3. Assert: ...
4. Cleanup: ...
```

---

## Files to Review

Start with these key files:

1. `tests/e2e/test_system_sanity.py` - Main E2E tests
2. `tests/conftest.py` - Fixtures and markers (lines 420-450 for E2E markers)
3. `kosmos/agents/research_director.py` - Main orchestrator (target for E2E)
4. `kosmos/core/research_loop.py` - Research loop (if exists)
5. `tests/integration/` - Integration tests that could be promoted to E2E

---

## Review Commands

Use these commands to gather information:

```bash
# Count E2E tests
pytest tests/e2e/ --collect-only 2>/dev/null | grep "test_" | wc -l

# List skipped tests
grep -r "@pytest.mark.skip" tests/e2e/

# Find tests requiring external deps
grep -r "requires_neo4j\|requires_api_key\|requires_docker" tests/

# Run E2E tests with verbose output (dry run)
pytest tests/e2e/ -v --collect-only

# Check test markers
grep -r "@pytest.mark.e2e" tests/
```

---

## Success Criteria

A good E2E test suite should:

1. **Cover the critical path**: Research question → validated hypothesis
2. **Test integration points**: All component boundaries exercised
3. **Handle failures gracefully**: Error paths tested, not just happy paths
4. **Run reliably**: No flaky tests, clear prerequisites
5. **Provide confidence**: Passing E2E tests mean the system works end-to-end
6. **Be maintainable**: Tests are readable, well-organized, and documented

---

## Output Format

Structure your review as:

```markdown
# E2E Testing Code Review: Kosmos

## Executive Summary
[2-3 sentences on overall state of E2E testing]

## Current Coverage
[Inventory of existing tests]

## Gap Analysis
[What's missing]

## Quality Issues
[Problems with existing tests]

## Recommendations
[Prioritized list of improvements]

## Appendix
[Detailed findings, test code examples]
```
