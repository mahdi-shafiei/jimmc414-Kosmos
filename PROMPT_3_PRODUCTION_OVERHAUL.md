# PROMPT 3: Production Overhaul - Tests, Documentation, and End-to-End Readiness

## Context

This prompt follows the completion of PROMPT 1 (Repository Setup) and PROMPT 2 (Gap Implementation). The Kosmos project now has:

- **6 critical gaps addressed** (5 fully implemented, 1 documented)
- **15 new files** with ~4,152 lines of production code
- **New modules**: `compression/`, `orchestration/`, `validation/`, `workflow/`
- **Reference arxiv paper**: `paper/2511.02824v2.pdf`

However, the test suite and documentation have not been updated to reflect these changes. This prompt guides a comprehensive overhaul to achieve production readiness.

---

## Objective

Perform a systematic overhaul to:
1. **Audit and update the test suite** - Add tests for new modules, deprecate obsolete tests
2. **Clean up documentation** - Archive stale documents, update README and guides
3. **Verify end-to-end workflow** - Ensure the system can operate as described in the arxiv paper
4. **Create production implementation plan** - Identify and document remaining work for full autonomous operation

---

## Phase 1: Test Suite Audit and Overhaul

### 1.1 New Modules Requiring Tests

The following modules were added in PR #16 but have **NO TESTS**:

```
kosmos/compression/
├── __init__.py
└── compressor.py (554 lines)

kosmos/orchestration/
├── __init__.py
├── delegation.py (468 lines)
├── novelty_detector.py (342 lines)
├── plan_creator.py (373 lines)
└── plan_reviewer.py (385 lines)

kosmos/validation/
├── __init__.py
└── scholar_eval.py (489 lines)

kosmos/workflow/
├── __init__.py
└── research_loop.py (375 lines)

kosmos/world_model/
└── artifacts.py (572 lines) - NEW

kosmos/agents/
└── skill_loader.py (429 lines) - NEW
```

### Task: Create Unit Tests

For each new module, create comprehensive unit tests:

1. **`tests/unit/compression/test_compressor.py`**
   - Test `ContextCompressor` class
   - Test `NotebookCompressor` class
   - Test `LiteratureCompressor` class
   - Test compression ratios and statistics extraction
   - Test lazy loading functionality
   - Mock LLM calls for deterministic testing

2. **`tests/unit/orchestration/test_delegation.py`**
   - Test task delegation logic
   - Test agent selection algorithms
   - Test parallel task distribution

3. **`tests/unit/orchestration/test_novelty_detector.py`**
   - Test novelty detection algorithms
   - Test duplicate detection
   - Test semantic similarity calculations

4. **`tests/unit/orchestration/test_plan_creator.py`**
   - Test plan generation from state
   - Test task prioritization
   - Test exploration vs exploitation balance

5. **`tests/unit/orchestration/test_plan_reviewer.py`**
   - Test plan validation
   - Test scientific soundness checks
   - Test feedback generation

6. **`tests/unit/validation/test_scholar_eval.py`**
   - Test 8-dimension evaluation framework
   - Test scoring calculations
   - Test threshold filtering

7. **`tests/unit/workflow/test_research_loop.py`**
   - Test cycle management
   - Test state transitions
   - Test termination conditions

8. **`tests/unit/world_model/test_artifacts.py`**
   - Test artifact creation and retrieval
   - Test JSON serialization
   - Test evidence chain tracking

9. **`tests/unit/agents/test_skill_loader.py`**
   - Test skill discovery from `kosmos-claude-scientific-skills/`
   - Test domain matching
   - Test prompt injection

### Task: Create Integration Tests

1. **`tests/integration/test_compression_pipeline.py`**
   - Test full compression workflow with real notebooks
   - Test multi-tier compression

2. **`tests/integration/test_orchestration_flow.py`**
   - Test plan creation -> review -> delegation pipeline
   - Test novelty detection integration

3. **`tests/integration/test_validation_pipeline.py`**
   - Test ScholarEval with real discoveries
   - Test filtering workflow

4. **`tests/integration/test_research_workflow.py`**
   - Test complete single-cycle workflow
   - Test state persistence across cycles

### Task: Create E2E Tests

1. **`tests/e2e/test_autonomous_research.py`**
   - Test multi-cycle autonomous operation (3-5 cycles)
   - Verify all components integrate correctly
   - Test against paper's described workflow

### 1.2 Tests to Review for Deprecation

Review these existing test directories for obsolete tests:

```
tests/unit/
├── agents/          - Check if tests cover new skill_loader
├── analysis/        - May need updates for new compression
├── core/            - Check workflow tests
├── execution/       - Review for Gap 4 changes
├── experiments/     - May be obsolete
├── hypothesis/      - Check integration with new orchestration
├── oversight/       - May need updates
├── world_model/     - Must add artifacts.py tests
```

**Criteria for deprecation:**
- Tests for removed/refactored code
- Tests that duplicate functionality
- Tests that no longer align with architecture
- Tests that can't pass due to architectural changes

### 1.3 Update Test Infrastructure

Update `tests/conftest.py` to add:
- Fixtures for new modules (compressor, orchestration, validation)
- Markers for Gap 4 dependent tests (require execution environment)
- Integration fixtures for multi-cycle testing

---

## Phase 2: Documentation Cleanup

### 2.1 Documents to Archive

Move to `archived/` directory:

```
archived/checkpoints/         - Already archived, verify complete
docs/planning/*.md            - Review each, archive obsolete planning docs
docs/phase-reports/           - Keep but mark as historical
IMPLEMENTATION_PLAN.md        - Review if still accurate
```

**Archive criteria:**
- Phase completion reports older than current implementation
- Checkpoint files superseded by newer versions
- Planning documents for completed work
- Resume prompts that are no longer needed

### 2.2 Documents to Update

1. **README.md**
   - Verify gap implementation status is accurate
   - Update test status badge
   - Add section on running the autonomous research loop
   - Update installation instructions if needed

2. **OPEN_QUESTIONS.md**
   - Mark addressed gaps as RESOLVED
   - Update with any new questions discovered

3. **IMPLEMENTATION_REPORT.md**
   - Add test coverage section
   - Add known limitations
   - Add performance benchmarks

4. **docs/user/user-guide.md**
   - Add guide for running autonomous research
   - Document new modules and their usage

5. **docs/developer/architecture.md**
   - Update architecture diagrams
   - Document new module relationships
   - Add compression and orchestration flows

### 2.3 Documents to Create

1. **docs/testing/TEST_GUIDE.md**
   - How to run tests
   - Test markers and categories
   - Writing new tests

2. **docs/operations/PRODUCTION_CHECKLIST.md**
   - Pre-production requirements
   - Environment setup
   - Monitoring recommendations

---

## Phase 3: End-to-End Verification

### 3.1 Paper Requirements Verification

Reference: `paper/2511.02824v2.pdf`

The paper describes Kosmos performing:
- Up to **20 research cycles**
- **12 hours maximum** runtime
- **10 parallel tasks** per cycle
- **200+ agent rollouts** per run
- Processing **1,500 papers**
- Generating **42,000 lines of code**

**Verification tasks:**

1. **Cycle Management**
   - [ ] Verify `research_loop.py` can manage 20 cycles
   - [ ] Verify cycle timing and termination conditions
   - [ ] Verify state persistence between cycles

2. **Parallel Execution**
   - [ ] Verify orchestration can dispatch 10 parallel tasks
   - [ ] Verify task result aggregation
   - [ ] Verify error handling for failed tasks

3. **Context Management**
   - [ ] Verify compression handles 1,500 papers
   - [ ] Verify compression handles 42,000 lines of code
   - [ ] Verify context fits within LLM limits

4. **State Manager Integration**
   - [ ] Verify artifacts.py integrates with existing world_model
   - [ ] Verify evidence chains are maintained
   - [ ] Verify queryable state for task generation

5. **Discovery Validation**
   - [ ] Verify ScholarEval scoring works end-to-end
   - [ ] Verify filtering removes low-quality discoveries
   - [ ] Verify traceability to evidence

### 3.2 Integration Points to Verify

```
┌─────────────────────────────────────────────────────────────────┐
│                     Research Loop (workflow/)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Plan Creator │───▶│ Plan Reviewer│───▶│  Delegation  │      │
│  │(orchestration)│   │(orchestration)│   │(orchestration)│     │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Artifacts  │◀───│  Compressor  │◀───│ Skill Loader │      │
│  │ (world_model)│    │(compression) │    │   (agents)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                       │               │
│         ▼                                       ▼               │
│  ┌──────────────┐                       ┌──────────────┐       │
│  │ Scholar Eval │                       │  Data Agent  │       │
│  │ (validation) │                       │  Lit Agent   │       │
│  └──────────────┘                       │  (existing)  │       │
│                                         └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

Verify each arrow represents a working integration.

---

## Phase 4: Production Implementation Plan

### 4.1 Gap 4 Completion (Execution Environment)

**Current Status**: Documented only, mock implementations

**Required for Production**:

1. **Docker-based Jupyter Kernel**
   ```
   kosmos/execution/
   ├── docker_kernel.py      - Docker container management
   ├── jupyter_executor.py   - Notebook execution
   ├── package_manager.py    - Dependency installation
   └── resource_limiter.py   - Memory/CPU/time limits
   ```

2. **Package Management**
   - Auto-detect required packages from generated code
   - Install packages in isolated environment
   - Handle version conflicts

3. **Multi-language Support**
   - Python execution (primary)
   - R execution via rpy2 (for MendelianRandomization, susieR)
   - Kernel switching based on task requirements

4. **Security**
   - Network isolation
   - Filesystem sandboxing
   - Resource limits (memory, CPU, disk, time)

### 4.2 Remaining Integration Work

1. **CLI Updates**
   - Add commands for autonomous research mode
   - Add progress monitoring
   - Add cycle inspection commands

2. **Configuration**
   - Add config options for:
     - Max cycles
     - Max runtime
     - Parallel task limit
     - Compression settings
     - Validation thresholds

3. **Monitoring**
   - Cycle progress tracking
   - Resource usage monitoring
   - Discovery count metrics
   - Quality score trends

### 4.3 Production Checklist

Before production deployment:

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] E2E test completes 5+ cycles successfully
- [ ] Documentation updated
- [ ] Stale docs archived
- [ ] Gap 4 execution environment implemented
- [ ] Security review completed
- [ ] Resource limits tested
- [ ] Error recovery tested
- [ ] Logging comprehensive
- [ ] Monitoring in place

---

## Execution Instructions

### Step 1: Run Current Tests
```bash
# Run existing tests to establish baseline
pytest tests/ -v --tb=short 2>&1 | tee test_baseline.log

# Count passing/failing
pytest tests/ --co -q | wc -l  # Total tests
```

### Step 2: Create Test Directories
```bash
mkdir -p tests/unit/compression
mkdir -p tests/unit/orchestration
mkdir -p tests/unit/validation
mkdir -p tests/unit/workflow
touch tests/unit/compression/__init__.py
touch tests/unit/orchestration/__init__.py
touch tests/unit/validation/__init__.py
touch tests/unit/workflow/__init__.py
```

### Step 3: Implement Tests (Priority Order)
1. Unit tests for `compressor.py` (foundational)
2. Unit tests for `artifacts.py` (state management)
3. Unit tests for `skill_loader.py` (agent integration)
4. Unit tests for orchestration modules
5. Unit tests for validation
6. Integration tests
7. E2E tests

### Step 4: Document as You Go
- Update IMPLEMENTATION_REPORT.md with test coverage
- Create TEST_GUIDE.md
- Archive obsolete docs

### Step 5: Verify End-to-End
```bash
# Run a short autonomous research test
python -m kosmos research --cycles 3 --max-time 1h --dataset sample.csv
```

---

## Success Criteria

This prompt is complete when:

1. **Tests**
   - [ ] New modules have >80% test coverage
   - [ ] All tests pass
   - [ ] Obsolete tests removed or updated

2. **Documentation**
   - [ ] Stale docs archived
   - [ ] README accurate
   - [ ] User guide updated
   - [ ] Architecture docs current

3. **End-to-End**
   - [ ] 3-cycle research workflow completes
   - [ ] All integration points verified
   - [ ] Paper requirements verified

4. **Production Plan**
   - [ ] Gap 4 implementation plan detailed
   - [ ] Production checklist complete
   - [ ] Remaining work documented

---

## Reference Files

- **Arxiv Paper**: `paper/2511.02824v2.pdf`
- **Gap Analysis**: `OPEN_QUESTIONS.md`
- **Implementation Report**: `IMPLEMENTATION_REPORT.md`
- **New Modules**:
  - `kosmos/compression/`
  - `kosmos/orchestration/`
  - `kosmos/validation/`
  - `kosmos/workflow/`
  - `kosmos/world_model/artifacts.py`
  - `kosmos/agents/skill_loader.py`
- **Reference Repos**: `kosmos-reference/`
- **Scientific Skills**: `kosmos-claude-scientific-skills/`
