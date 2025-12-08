# Paper Implementation Gaps

**Document Purpose**: Track gaps between the original Kosmos paper claims and this implementation.
**Paper Reference**: Mitchener et al., "Kosmos: An AI Scientist for Autonomous Discovery" (arXiv:2511.02824v2)
**Last Updated**: 2025-12-08

---

## Summary

| Priority | Count | Status |
|----------|-------|--------|
| Critical | 5 | 0/5 Complete |
| High | 3 | 0/3 Complete |
| Medium | 2 | 0/2 Complete |
| Low | 2 | 0/2 Complete |
| **Total** | **12** | **0/12 Complete** |

---

## Critical Priority Gaps

### GAP-001: Self-Correcting Code Execution

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#54](https://github.com/jimmc414/Kosmos/issues/54) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | Execution |

**Paper Claim** (Section 4, Phase B):
> "Data Analysis Agent: If error occurs → reads traceback → fixes code → re-executes (iterative debugging)"

**Current Implementation**:
- Traceback capture exists: `kosmos/execution/executor.py:256`
- Basic retry logic with `max_retries=3`
- `RetryStrategy.modify_code_for_retry()` only handles `KeyError` and `FileNotFoundError`
- Returns `None` for all other error types

**Gap**:
- No intelligent code repair
- No LLM-based analysis of tracebacks
- Most errors trigger retry without fixing the underlying issue

**Files to Modify**:
- `kosmos/execution/executor.py:436-520`

**Acceptance Criteria**:
- [ ] RetryStrategy handles >5 common error types
- [ ] LLM analyzes traceback and suggests fix
- [ ] Code is modified before retry attempt
- [ ] Success rate tracked for auto-repairs

---

### GAP-002: World Model Update Categories

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#55](https://github.com/jimmc414/Kosmos/issues/55) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | World Model |

**Paper Claim** (Section 4, Phase C):
> "The World Model integrates findings using three categories: **Confirmation** (data supports hypothesis), **Conflict** (data contradicts literature), **Pruning** (hypothesis refuted)"

**Current Implementation**:
- `add_finding_with_conflict_check()` exists at `kosmos/world_model/artifacts.py:507-531`
- Function is a **STUB** that always returns `False` for conflicts
- Comment says "Future: More sophisticated conflict detection"
- No enum for update types

**Gap**:
- No Confirmation/Conflict/Pruning categorization
- No semantic conflict detection
- No hypothesis pruning workflow

**Files to Modify**:
- `kosmos/world_model/artifacts.py`
- `kosmos/core/workflow.py` (state transitions for pruning)

**Acceptance Criteria**:
- [ ] `UpdateType` enum with CONFIRMATION, CONFLICT, PRUNING
- [ ] Conflict detection using semantic similarity
- [ ] Pruned hypotheses marked and excluded from future cycles
- [ ] Statistics tracked for each update type

---

### GAP-003: 12-Hour Runtime Constraint

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#56](https://github.com/jimmc414/Kosmos/issues/56) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | Configuration |

**Paper Claim** (Section 1):
> "Standard Runtime: Up to 12 hours (continuous operation)"

**Current Implementation**:
- No `MAX_RUNTIME`, `RUNTIME_LIMIT`, or similar in code
- Only cycle limits (`max_iterations=20`)
- Task-level timeouts (60-600s) but no global timeout

**Gap**:
- System could run indefinitely
- No graceful shutdown at time limit
- No elapsed time tracking

**Files to Modify**:
- `kosmos/config.py` (add MAX_RUNTIME_HOURS)
- `kosmos/workflow/research_loop.py` (check elapsed time)

**Acceptance Criteria**:
- [ ] `MAX_RUNTIME_HOURS` config option (default: 12)
- [ ] Elapsed time tracked from run start
- [ ] Graceful shutdown when limit approached
- [ ] Final report generated before timeout

---

### GAP-004: Parallel Task Execution Mismatch

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#57](https://github.com/jimmc414/Kosmos/issues/57) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | Configuration |

**Paper Claim** (Section 4, Phase A):
> "Generates a batch of **up to 10 parallel tasks**"

**Current Implementation**:
- Task generation: 10 tasks per cycle (`tasks_per_cycle=10`) ✓
- Execution: Default `max_concurrent_experiments=4` at `config.py:708`
- Maximum configurable: 16

**Gap**:
- Paper claims 10 parallel, implementation defaults to 4
- 2.5x lower throughput than paper claims

**Files to Modify**:
- `kosmos/config.py:708`

**Acceptance Criteria**:
- [ ] Default `max_concurrent_experiments=10`
- [ ] Documentation updated to reflect parallel capacity
- [ ] Performance tested at 10 concurrent

---

### GAP-005: Agent Rollout Tracking

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#58](https://github.com/jimmc414/Kosmos/issues/58) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | Agents |

**Paper Claim** (Section 1):
> "Agent Rollouts: ~200 total (~166 data analysis, ~36 literature)"

**Current Implementation**:
- `strategy_stats` tracks "attempts" at `research_director.py:136-141`
- No breakdown by agent type (data vs literature)
- No "rollout" terminology or counting

**Gap**:
- Cannot verify operational scale matches paper
- No metrics for agent-typed rollouts

**Files to Modify**:
- `kosmos/agents/research_director.py`
- `kosmos/core/metrics.py`

**Acceptance Criteria**:
- [ ] `RolloutTracker` class with per-agent-type counts
- [ ] Summary shows "X data analysis + Y literature rollouts"
- [ ] Rollout count included in final report

---

## High Priority Gaps

### GAP-006: h5ad/Parquet Data Format Support

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#59](https://github.com/jimmc414/Kosmos/issues/59) |
| **Status** | Not Started |
| **Priority** | High |
| **Area** | Execution |

**Paper Claim** (Section 3.1):
> "Format: CSV, TSV, Parquet, Excel, or scientific formats (e.g., h5ad for single-cell RNA-seq)"

**Current Implementation**:
- CSV: `DataLoader.load_csv()` at `data_analysis.py:832` ✓
- Excel: `DataLoader.load_excel()` at line 849 ✓
- h5ad: Not implemented
- Parquet: Not implemented

**Gap**:
- Cannot process single-cell RNA-seq datasets (h5ad is standard)
- Cannot process columnar analytics data (Parquet)

**Files to Modify**:
- `kosmos/execution/data_analysis.py`
- `requirements.txt` (add anndata, pyarrow)

**Acceptance Criteria**:
- [ ] `DataLoader.load_h5ad()` using anndata library
- [ ] `DataLoader.load_parquet()` using pyarrow
- [ ] Auto-detection by file extension
- [ ] Tests with real h5ad/parquet files

---

### GAP-007: Figure Generation

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#60](https://github.com/jimmc414/Kosmos/issues/60) |
| **Status** | Not Started |
| **Priority** | High |
| **Area** | Execution |

**Paper Claim** (Section 5):
> "High-resolution figures: High-resolution plots generated by the Data Analysis Agent"

**Current Implementation**:
- No matplotlib/seaborn imports in execution code
- No `plt.savefig()` or figure export
- Code templates have comments like `"# Plot results"` but no actual code

**Gap**:
- System cannot produce visual outputs
- Reports lack figures despite paper claim

**Files to Modify**:
- `kosmos/execution/code_generator.py`
- `kosmos/execution/templates/` (add figure templates)

**Acceptance Criteria**:
- [ ] Code templates include matplotlib plotting
- [ ] Figures saved to `artifacts/cycle_N/figures/`
- [ ] High-resolution export (300+ DPI)
- [ ] Figure references in reports

---

### GAP-008: Jupyter Notebook Generation

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#61](https://github.com/jimmc414/Kosmos/issues/61) |
| **Status** | Not Started |
| **Priority** | High |
| **Area** | Execution |

**Paper Claim** (Section 5):
> "Code Repository: All ~42,000 lines of executable Python code generated during the run (Jupyter notebooks)"

**Current Implementation**:
- `JupyterClient` can EXECUTE notebooks at `jupyter_client.py:326`
- Cannot CREATE notebooks from code
- Compression processes existing notebooks but doesn't generate them

**Gap**:
- System doesn't produce notebook artifacts as claimed
- Code not preserved in reproducible format

**Files to Modify**:
- `kosmos/execution/jupyter_client.py`
- New: `kosmos/execution/notebook_generator.py`

**Acceptance Criteria**:
- [ ] `NotebookGenerator.create_notebook(code, outputs)` function
- [ ] Notebooks saved to `artifacts/cycle_N/notebooks/`
- [ ] Outputs embedded in notebook cells
- [ ] Total line count tracked

---

## Medium Priority Gaps

### GAP-009: Code Line Provenance

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#62](https://github.com/jimmc414/Kosmos/issues/62) |
| **Status** | Not Started |
| **Priority** | Medium |
| **Area** | Traceability |

**Paper Claim** (Section 5):
> "Code Citation: Hyperlink to the exact Jupyter notebook and line of code that produced the claim"

**Current Implementation**:
- DOI support for literature citations ✓
- No code line → finding mapping
- Phase 4 doc says "PROV-O provenance tracking" is future work

**Gap**:
- Cannot audit which code line produced which finding
- No hyperlinks to source code in reports

**Files to Modify**:
- `kosmos/world_model/artifacts.py`
- `kosmos/execution/executor.py`

**Acceptance Criteria**:
- [ ] Findings include `source_file` and `line_number` fields
- [ ] Report generator creates hyperlinks to code
- [ ] Provenance chain: finding → code → hypothesis

---

### GAP-010: Failure Mode Detection

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#63](https://github.com/jimmc414/Kosmos/issues/63) |
| **Status** | Not Started |
| **Priority** | Medium |
| **Area** | Validation |

**Paper Claim** (Section 6.2):
> "Common failure modes: Over-interpretation, Invented Metrics, Pipeline Pivots, Rabbit Holes"

**Current Implementation**:
- Loop prevention: `MAX_ACTIONS_PER_ITERATION=50` ✓
- Error recovery: `MAX_CONSECUTIVE_ERRORS=3` ✓
- No over-interpretation detection
- No invented metrics validation
- No rabbit hole prevention

**Gap**:
- System may make speculative claims without flagging
- May report non-existent metrics
- May explore irrelevant tangents

**Files to Modify**:
- `kosmos/validation/scholar_eval.py`
- `kosmos/core/convergence.py`

**Acceptance Criteria**:
- [ ] Confidence score for interpretations vs facts
- [ ] Validation that claimed metrics exist in data
- [ ] Relatedness check to original research question
- [ ] Warnings for potential failure modes

---

## Low Priority Gaps

### GAP-011: Multi-Run Convergence Framework

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#64](https://github.com/jimmc414/Kosmos/issues/64) |
| **Status** | Not Started |
| **Priority** | Low |
| **Area** | Workflow |

**Paper Claim** (Section 6.3):
> "Kosmos is non-deterministic. If a finding is critical, run multiple times and look for convergent results."

**Current Implementation**:
- Temperature control exists (0.0-0.7)
- No framework for N independent runs
- No ensemble averaging
- No convergence metrics

**Gap**:
- Cannot validate findings through replication
- No confidence from multiple runs

**Files to Modify**:
- `kosmos/workflow/research_loop.py`
- New: `kosmos/workflow/ensemble.py`

**Acceptance Criteria**:
- [ ] `EnsembleRunner.run(n_runs, research_objective)` function
- [ ] Convergence metrics across runs
- [ ] Report showing findings that appeared in N/M runs

---

### GAP-012: Paper Accuracy Validation

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#65](https://github.com/jimmc414/Kosmos/issues/65) |
| **Status** | Not Started |
| **Priority** | Low |
| **Area** | Validation |

**Paper Claim** (Section 8):
> "79.4% overall accuracy, 85.5% data analysis, 82.1% literature, 57.9% interpretation"

**Current Implementation**:
- ScholarEval framework exists ✓
- Test framework with accuracy targets defined ✓
- No validation study conducted
- `120625_code_review.md` says "Paper claims NOT yet reproduced"

**Gap**:
- Cannot verify system achieves paper accuracy
- No benchmark dataset for validation

**Files to Modify**:
- `tests/requirements/scientific/test_req_sci_validation.py`
- New: benchmark dataset

**Acceptance Criteria**:
- [ ] Validation study with expert-annotated dataset
- [ ] Accuracy measured by statement type
- [ ] Results compared to paper claims
- [ ] Report documenting any deviations

---

## What IS Correctly Implemented

| Feature | Paper Claim | Status |
|---------|-------------|--------|
| 20 research cycles | Up to 20 cycles | ✅ `max_iterations=20` configurable |
| Literature APIs | PubMed, arXiv, Semantic Scholar | ✅ All three with ThreadPoolExecutor |
| ScholarEval validation | 8-dimension peer review | ✅ Fully implemented |
| Context compression | 20:1 compression ratio | ✅ Hierarchical summarization |
| Plan Creator + Reviewer | Task generation with QA | ✅ 5-dimension review scoring |
| Convergence detection | Multiple stopping criteria | ✅ 8 criteria implemented |
| Budget tracking | Cost enforcement | ✅ Graceful convergence |

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-08 | Initial document created with 12 gaps identified |
