# Issue #6 Analysis Report: Kosmos Stalls During Autonomous Research with Ollama

**Issue:** Bug Report: Kosmos stalls during Autonomous Research when using Ollama
**Reporter:** dangkhoa2250
**Reported:** November 18, 2025
**Latest Update:** November 24, 2025
**Status:** OPEN
**Version:** Kosmos v0.2.0+

---

## Executive Summary

When running Kosmos with Ollama as the LLM provider, the system encounters multiple failure modes:

1. **Response parsing failures** - Kosmos cannot correctly parse Ollama's output format, especially for structured JSON responses
2. **Retry storm problem** - Failed parsing triggers repeated requests rather than graceful error handling
3. **Resource exhaustion** - The retry loops eventually exhaust Ollama's resources, causing service shutdown
4. **Cascading failure** - Latest error: `'NoneType' object has no attribute 'title'`

This analysis identifies **7 distinct issues** across **9 files** that need to be addressed.

---

## Root Cause Analysis

### Primary Issue 1: JSON Parsing Failures with Local Models

Local models like Ollama often produce responses that don't strictly follow JSON formatting instructions. The current parsing implementation in multiple files is not robust enough.

**Affected Files:**

| File | Line(s) | Issue |
|------|---------|-------|
| `kosmos/core/providers/openai.py` | 388-404 | JSON parsing with minimal error recovery |
| `kosmos/core/llm.py` | 439-455 | Similar JSON parsing issues |
| `kosmos/core/providers/anthropic.py` | 426-442 | Same pattern |

**Current Code (`kosmos/core/providers/openai.py:388-404`):**
```python
# Parse JSON (handle markdown code blocks)
try:
    if "```json" in response_text:
        json_start = response_text.find("```json") + 7
        json_end = response_text.find("```", json_start)
        response_text = response_text[json_start:json_end].strip()
    elif "```" in response_text:
        json_start = response_text.find("```") + 3
        json_end = response_text.find("```", json_start)
        response_text = response_text[json_start:json_end].strip()

    return json.loads(response_text)

except json.JSONDecodeError as e:
    logger.error(f"Failed to parse JSON: {e}")
    logger.error(f"Response text: {response_text[:500]}")
    raise ProviderAPIError("openai", f"Invalid JSON response: {e}", raw_error=e)
```

**Problems:**
1. No attempt to extract JSON from mixed text/JSON responses
2. No handling of common local model quirks (extra whitespace, trailing commas, single quotes)
3. Throws exception immediately instead of attempting recovery
4. No model-specific prompt adjustments for JSON generation

### Primary Issue 2: Missing Null Checks for Paper Objects

The `'NoneType' object has no attribute 'title'` error indicates that code is accessing `.title` on objects that can be `None`. This happens in literature search results where a paper object may be null.

**Affected Files:**

| File | Line(s) | Issue |
|------|---------|-------|
| `kosmos/agents/hypothesis_generator.py` | 322-326 | Paper title access without null check |
| `kosmos/agents/hypothesis_generator.py` | 380 | List comprehension with potential null objects |
| `kosmos/hypothesis/novelty_checker.py` | 144, 323, 327 | Paper title access in similarity calculations |
| `kosmos/agents/literature_analyzer.py` | 286, 529, 726, 767 | Multiple paper title accesses |

**Current Code (`kosmos/agents/hypothesis_generator.py:322-326`):**
```python
if context_papers:
    literature_context = "Recent relevant literature:\n\n"
    for i, paper in enumerate(context_papers[:5], 1):
        literature_context += f"{i}. {paper.title} ({paper.year})\n"  # FAILS if paper is None
        if paper.abstract:
            literature_context += f"   Abstract: {paper.abstract[:200]}...\n"
```

**Problem:** If any paper in `context_papers` is `None`, accessing `paper.title` raises `AttributeError`.

### Primary Issue 3: Aggressive Retry Logic Without Resource Awareness

The retry mechanisms don't account for local model resource constraints. When Ollama fails due to parsing issues, the retry logic sends more requests, compounding the problem.

**Affected Files:**

| File | Line(s) | Issue |
|------|---------|-------|
| `kosmos/core/async_llm.py` | 271-307 | Retry logic with exponential backoff up to 30s wait |
| `kosmos/orchestration/delegation.py` | 278-340 | Task retry with max 2 attempts |
| `kosmos/execution/executor.py` | 145-167, 427-466 | Code execution retry |
| `kosmos/config.py` | 252 | `retry_on_timeout` config flag |

**Current Retry Logic (`kosmos/core/async_llm.py:274-295`):**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
)
async def _call_api():
    return await asyncio.wait_for(
        self.client.messages.create(...),
        timeout=timeout
    )
```

**Problems:**
1. Retry on `APIError` includes JSON parsing failures (wrapped in ProviderAPIError)
2. No differentiation between recoverable errors (network) and non-recoverable (bad response format)
3. Local models may not recover from repeated identical requests
4. No circuit breaker pattern to prevent resource exhaustion

### Primary Issue 4: Insufficient Local Model Support in generate_structured()

The `generate_structured()` method relies heavily on the model following JSON schema instructions. Local models are less reliable at this task.

**Current Flow:**
```
1. User calls generate_structured(prompt, schema)
2. System adds JSON instruction to system prompt
3. Model generates response (may not be valid JSON)
4. json.loads() fails
5. ProviderAPIError raised
6. Retry logic kicks in
7. Same request, same failure
8. Loop until max retries exceeded
```

**Missing:**
- Schema validation relaxation for local models
- Alternative structured output strategies (e.g., key-value extraction)
- Model-specific prompt engineering for JSON output
- Graceful degradation to unstructured output with post-processing

---

## Detailed Impact Analysis

### Impact on Research Workflow

When Ollama is used during autonomous research, the following workflow stages are affected:

| Stage | Component | Failure Mode |
|-------|-----------|--------------|
| **Hypothesis Generation** | `HypothesisGeneratorAgent._generate_with_claude()` | JSON parsing failure for hypothesis schema |
| **Literature Search** | `UnifiedLiteratureSearch.search()` | Returns `None` papers in some cases |
| **Context Building** | `HypothesisGeneratorAgent._gather_literature_context()` | NoneType error when iterating papers |
| **Novelty Checking** | `NoveltyChecker._calculate_semantic_similarity()` | Title access on None paper |
| **Domain Detection** | `HypothesisGeneratorAgent._detect_domain()` | May fail on non-standard responses |
| **Experiment Design** | Research workflow | JSON schema requirements not met |
| **Result Analysis** | DataAnalystAgent | Structured output parsing failures |

### Resource Exhaustion Timeline

Based on the code analysis, here's the typical failure progression:

```
T+0s:   First request to Ollama (generate_structured)
T+2s:   JSON parsing fails, retry #1 initiated
T+4s:   Same failure, retry #2 initiated
T+8s:   Same failure, retry #3 initiated
T+16s:  Max retries exceeded, ProviderAPIError propagated
T+16s:  Orchestration layer catches error
T+16s:  Orchestration retry #1 (new request)
T+32s:  Orchestration retry #2
T+48s:  Error propagated to ResearchDirector
T+48s:  ResearchDirector may initiate new cycle
...repeat...
```

With default settings: **up to 9 retry attempts** (3 provider × 2 orchestration + 3 executor)

---

## Recommended Changes

### Fix 1: Robust JSON Parsing with Fallback Strategies

**File:** `kosmos/core/providers/openai.py` (and similar in `anthropic.py`, `llm.py`)

**Current:** Lines 388-404

**Proposed:**

```python
def _parse_json_response(self, response_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse JSON from model response with multiple fallback strategies.

    Strategies tried in order:
    1. Direct JSON parse
    2. Extract from markdown code blocks
    3. Extract JSON object using regex
    4. Extract key-value pairs and construct JSON
    5. Return partial/default schema on failure
    """
    import re

    # Strategy 1: Direct parse
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        r'\{[\s\S]*\}',                   # Any JSON object
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                # Clean common issues
                json_str = json_str.strip()
                json_str = re.sub(r',\s*}', '}', json_str)  # Trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Trailing commas in arrays
                json_str = json_str.replace("'", '"')        # Single quotes to double
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    # Strategy 3: For local models, try key-value extraction
    if self.provider_type in ('local', 'compatible'):
        try:
            return self._extract_key_values(response_text, schema)
        except Exception:
            pass

    # Log failure details
    logger.warning(f"JSON parsing failed for response: {response_text[:500]}")
    logger.warning(f"Expected schema: {schema}")

    # Strategy 4: Return partial with defaults for non-critical fields
    raise ProviderAPIError(
        "openai",
        f"Could not parse JSON response. Model output: {response_text[:200]}...",
        recoverable=False  # Don't retry same request
    )

def _extract_key_values(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key-value pairs from unstructured text based on schema."""
    result = {}
    for key, expected_type in schema.items():
        # Try to find key: value or "key": value patterns
        patterns = [
            rf'"{key}"\s*:\s*"([^"]*)"',      # "key": "value"
            rf'"{key}"\s*:\s*(\d+\.?\d*)',     # "key": number
            rf'{key}\s*:\s*([^\n,}}]+)',        # key: value (loose)
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
                break
    return result if result else None
```

### Fix 2: Null-Safe Paper Object Access

**File:** `kosmos/agents/hypothesis_generator.py`

**Current (lines 322-326):**
```python
if context_papers:
    literature_context = "Recent relevant literature:\n\n"
    for i, paper in enumerate(context_papers[:5], 1):
        literature_context += f"{i}. {paper.title} ({paper.year})\n"
```

**Proposed:**
```python
if context_papers:
    literature_context = "Recent relevant literature:\n\n"
    valid_papers = [p for p in context_papers[:5] if p is not None and p.title]
    for i, paper in enumerate(valid_papers, 1):
        title = paper.title or "Untitled"
        year = paper.year or "N/A"
        literature_context += f"{i}. {title} ({year})\n"
        if paper.abstract:
            literature_context += f"   Abstract: {paper.abstract[:200]}...\n"
```

**File:** `kosmos/agents/hypothesis_generator.py`

**Current (line 380):**
```python
related_papers=[p.arxiv_id or p.doi or p.title for p in context_papers if p.arxiv_id or p.doi]
```

**Proposed:**
```python
related_papers=[
    p.arxiv_id or p.doi or p.title
    for p in context_papers
    if p is not None and (p.arxiv_id or p.doi or p.title)
]
```

### Fix 3: Add Non-Recoverable Error Classification

**File:** `kosmos/core/providers/base.py`

**Add to ProviderAPIError class:**
```python
class ProviderAPIError(Exception):
    """Exception raised when a provider API call fails."""

    def __init__(
        self,
        provider: str,
        message: str,
        raw_error: Optional[Exception] = None,
        recoverable: bool = True  # NEW: Flag for retry decisions
    ):
        self.provider = provider
        self.message = message
        self.raw_error = raw_error
        self.recoverable = recoverable  # NEW
        super().__init__(f"{provider}: {message}")

    def is_recoverable(self) -> bool:
        """Check if this error is recoverable through retry."""
        if not self.recoverable:
            return False

        # Network/timeout errors are recoverable
        recoverable_types = (
            'timeout', 'connection', 'network',
            'rate_limit', 'overloaded', 'service_unavailable'
        )
        message_lower = self.message.lower()
        return any(t in message_lower for t in recoverable_types)
```

### Fix 4: Smart Retry Logic with Circuit Breaker

**File:** `kosmos/core/async_llm.py` (and similar for sync code)

**Proposed Enhancement:**
```python
class CircuitBreaker:
    """Circuit breaker pattern for provider calls."""

    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def can_proceed(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        # half-open: allow one request to test
        return True

# Usage in retry logic:
def should_retry(self, error: Exception, attempt: int) -> bool:
    """Determine if request should be retried."""
    if isinstance(error, ProviderAPIError):
        if not error.is_recoverable():
            logger.info(f"Non-recoverable error, skipping retry: {error}")
            return False

    if not self.circuit_breaker.can_proceed():
        logger.warning("Circuit breaker open, skipping retry")
        return False

    return attempt < self.max_retries
```

### Fix 5: Local Model-Specific Configuration

**File:** `kosmos/config.py`

**Add new configuration section:**
```python
class LocalModelConfig(BaseSettings):
    """Configuration for local models (Ollama, LM Studio, etc.)."""

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_MODEL_",
        extra="ignore"
    )

    # Retry configuration
    max_retries: int = Field(
        default=1,  # Lower for local models
        description="Maximum retry attempts for local models"
    )

    # JSON parsing
    strict_json: bool = Field(
        default=False,
        description="Require strict JSON compliance (False for local models)"
    )

    json_retry_with_hint: bool = Field(
        default=True,
        description="On JSON parse failure, retry with explicit formatting hint"
    )

    # Resource management
    request_timeout: int = Field(
        default=120,
        description="Timeout for local model requests (seconds)"
    )

    concurrent_requests: int = Field(
        default=1,
        description="Max concurrent requests to local model"
    )

    # Graceful degradation
    fallback_to_unstructured: bool = Field(
        default=True,
        description="On structured output failure, try unstructured extraction"
    )
```

### Fix 6: Enhanced Error Messages for Local Models

**File:** `kosmos/core/providers/openai.py`

**Add after line 237:**
```python
except Exception as e:
    error_msg = str(e)

    # Provide helpful guidance for local model issues
    if self.provider_type == 'local':
        if 'json' in error_msg.lower() or 'parse' in error_msg.lower():
            logger.error(
                f"JSON parsing failed with local model ({self.model}). "
                f"Local models may not reliably produce structured JSON output. "
                f"Consider:\n"
                f"  1. Using a larger model (e.g., llama3.1:70b instead of :8b)\n"
                f"  2. Setting LOCAL_MODEL_STRICT_JSON=false\n"
                f"  3. Using a cloud provider for complex structured outputs"
            )
        elif 'timeout' in error_msg.lower():
            logger.error(
                f"Request to local model timed out. "
                f"Consider increasing LOCAL_MODEL_REQUEST_TIMEOUT or using a smaller model."
            )

    raise ProviderAPIError("openai", f"Generation failed: {e}", raw_error=e)
```

### Fix 7: Literature Search Null Handling

**File:** `kosmos/literature/unified_search.py`

**Add validation in search method after line 160:**
```python
for future in as_completed(future_to_source):
    source = future_to_source[future]
    try:
        papers = future.result()
        # Filter out None values and validate paper objects
        valid_papers = [
            p for p in papers
            if p is not None and hasattr(p, 'title') and p.title
        ]
        if len(valid_papers) < len(papers):
            logger.warning(
                f"Filtered {len(papers) - len(valid_papers)} invalid papers from {source.value}"
            )
        all_papers.extend(valid_papers)
        logger.info(f"Retrieved {len(valid_papers)} valid papers from {source.value}")
    except Exception as e:
        logger.error(f"Error searching {source.value}: {e}")
```

---

## Files to Modify Summary

| File | Priority | Changes Required |
|------|----------|-----------------|
| `kosmos/core/providers/openai.py` | **Critical** | Robust JSON parsing, error classification, local model support |
| `kosmos/core/providers/base.py` | **Critical** | Add `recoverable` flag to ProviderAPIError |
| `kosmos/agents/hypothesis_generator.py` | **Critical** | Null-safe paper access (lines 322-326, 380) |
| `kosmos/core/async_llm.py` | **High** | Circuit breaker, smart retry logic |
| `kosmos/literature/unified_search.py` | **High** | Filter invalid papers from results |
| `kosmos/hypothesis/novelty_checker.py` | **High** | Null-safe paper title access |
| `kosmos/config.py` | **Medium** | LocalModelConfig section |
| `kosmos/core/providers/anthropic.py` | **Medium** | Same JSON parsing improvements |
| `kosmos/core/llm.py` | **Medium** | Same JSON parsing improvements |

---

## Testing Strategy

### Unit Tests Required

1. **JSON Parsing Robustness**
   ```python
   def test_parse_json_with_markdown_blocks():
       """Test JSON extraction from various markdown formats."""

   def test_parse_json_with_trailing_commas():
       """Test handling of malformed JSON from local models."""

   def test_parse_json_fallback_to_keyvalue():
       """Test key-value extraction when JSON parse fails."""
   ```

2. **Null Paper Handling**
   ```python
   def test_hypothesis_generation_with_none_papers():
       """Test hypothesis generator handles None papers gracefully."""

   def test_literature_context_with_partial_papers():
       """Test context building with papers missing title/year."""
   ```

3. **Retry Logic**
   ```python
   def test_no_retry_on_non_recoverable_error():
       """Test that JSON parse errors don't trigger infinite retries."""

   def test_circuit_breaker_opens_after_failures():
       """Test circuit breaker prevents resource exhaustion."""
   ```

### Integration Tests

1. **Ollama End-to-End**
   - Configure OpenAI provider with Ollama
   - Run simple generation
   - Run structured generation
   - Verify graceful handling of parse failures

2. **Autonomous Research with Local Model**
   - Run short research cycle (max 2 iterations)
   - Verify no infinite loops
   - Verify proper error messages

### Manual Test Checklist

```bash
# 1. Verify Ollama compatibility
python tests/manual/test_ollama.py

# 2. Run doctor with Ollama config
export LLM_PROVIDER=openai
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama3.1:8b
kosmos doctor

# 3. Test structured output with local model
kosmos test-llm --structured

# 4. Short research run
kosmos run --question "What is machine learning?" --max-iterations 1
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing Anthropic/OpenAI usage | Low | High | Extensive test coverage, backward compatibility checks |
| JSON parsing changes affect valid responses | Medium | Medium | Preserve strict parsing first, fallback only on failure |
| Circuit breaker too aggressive | Low | Medium | Configurable thresholds, start conservative |
| Local model users expect cloud-level quality | High | Low | Clear documentation, helpful error messages |

---

## Implementation Order

1. **Phase 1 (Critical - Immediate)**
   - Fix null paper access in `hypothesis_generator.py`
   - Add `recoverable` flag to `ProviderAPIError`
   - Update retry logic to check recoverability

2. **Phase 2 (High - Short term)**
   - Implement robust JSON parsing with fallbacks
   - Add circuit breaker to async_llm
   - Filter invalid papers in unified_search

3. **Phase 3 (Medium - Planned)**
   - Add LocalModelConfig
   - Enhanced error messages
   - Comprehensive test suite

---

## Conclusion

Issue #6 stems from multiple compounding problems when using Ollama:

1. **Immediate cause**: `'NoneType' object has no attribute 'title'` - papers can be None
2. **Secondary cause**: JSON parsing failures with local models trigger retries
3. **Systemic cause**: Retry logic doesn't distinguish recoverable from non-recoverable errors
4. **Resource impact**: Multiple retry layers compound, exhausting Ollama

The recommended fixes address all layers:
- Defensive coding for null objects
- Robust JSON parsing with multiple fallback strategies
- Smart retry logic with circuit breaker
- Local model-specific configuration and guidance

**Estimated LOC Changes:** ~400 lines across 9 files
**Estimated Testing:** ~200 lines of new tests
**Backward Compatibility:** Maintained through default configurations

---

*Report generated: 2025-11-26*
*Analyzed by: Claude Code Analysis*
*Related Issues: #22 (ANTHROPIC_API_KEY requirement - fixed)*
