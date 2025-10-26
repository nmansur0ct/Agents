<<<<<<< HEAD
# Technical Deep Dive: Agentic Implementation Details

## Enhanced Agentic Prompt Architecture

### Agent Identity and Role Definition

The system now implements **proper agentic identity** where each LLM interaction begins with explicit agent role definition:

```python
# Feature Analysis Agent
prompt = f"""You are an agent for feature analysis and factor assessment. Your role is to analyze product features and estimate normalized business factors.

From the feature description below, estimate normalized factors in [0,1] range:
- reach: How many users/customers will this impact?
- revenue: How much will this contribute to revenue growth?
- risk_reduction: How much does this reduce business/technical risk?
- engineering: How much engineering effort is required?
- dependency: How many external dependencies/integrations?
- complexity: How complex is the implementation?

Feature Description: {feature_desc}

Return STRICT JSON with keys: reach, revenue, risk_reduction, engineering, dependency, complexity, notes (array of short analysis strings)."""

# Business Rationale Agent  
prompt = f"""You are an agent for business rationale generation. Your role is to create clear, actionable business justifications for feature prioritization decisions.

Analyze this feature's priority score and generate a concise business rationale..."""
```

### Key Benefits of Agentic Prompts

1. **Clear Role Boundaries**: Each agent knows its specific function and domain
2. **Consistent Behavior**: Agents maintain character across different feature types
3. **Structured Output**: Simplified JSON format focusing on core factors
4. **Enhanced Reasoning**: Agents understand they are autonomous decision-makers
5. **Improved Reliability**: More predictable responses due to clear expectations

## Agent Architecture Patterns

### 1. State-Based Agent Communication

The system implements the **Shared State Pattern** where agents communicate through a common state object:

```python
class State(TypedDict):
    raw: List[RawFeature]           # Input data
    extracted: ExtractorOutput     # Agent 1 → Agent 2
    scored: ScorerOutput          # Agent 2 → Agent 3  
    prioritized: PrioritizedOutput # Agent 3 → Output
    errors: List[str]             # Cross-agent error tracking
```

**Benefits:**
- **Loose Coupling**: Agents don't directly depend on each other
- **Auditability**: Complete state history for debugging
- **Extensibility**: Easy to add new agents without changing existing ones

### 2. Agent Composition Pattern

Each agent is implemented as a pure function with dependency injection:

```python
def extractor_node(state: State, config: Config) -> State:
    """
    Agent implementation as pure function:
    - Immutable inputs (state, config)
    - Predictable outputs (updated state)
    - No side effects (except logging)
    """
    extracted_features = []
    
    for raw_feature in state.get('raw', []):
        # Agent decision-making logic
        feature_spec = analyze_and_extract(raw_feature, config)
        extracted_features.append(feature_spec)
    
    # Return updated state
    return {**state, 'extracted': ExtractorOutput(features=extracted_features)}
```

### 3. LLM Integration Pattern

Agents use the **Enhancement Decorator Pattern** for LLM integration:

```python
def _enhance_with_llm(feature_desc: str, config: Config) -> Dict[str, Any]:
    """
    LLM enhancement layer that can be toggled on/off
    """
    if not config.llm_enabled:
        return {}  # Graceful degradation
    
    try:
        # Structured prompt engineering
        response = call_openai_json(
            model=config.llm_model,
            prompt=construct_analysis_prompt(feature_desc),
            temperature=config.llm_temperature
        )
        return response
    except Exception:
        # Transparent fallback
        print("⚠️ LLM enhancement failed, using heuristics")
        return {}
```

## LangGraph Orchestration Details

### Graph Compilation and Execution

```python
def _build_graph(self):
    """
    LangGraph provides declarative workflow definition
    """
    workflow = StateGraph(State)
    
    # Add autonomous agents as nodes
    workflow.add_node("extract", extractor_with_config)
    workflow.add_node("score", scorer_with_config)
    workflow.add_node("prioritize", prioritizer_with_config)
    
    # Define agent execution flow
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "score")      # Sequential execution
    workflow.add_edge("score", "prioritize")
    workflow.add_edge("prioritize", END)
    
    return workflow.compile()  # Creates executable graph
```

### Agent Lifecycle Management

1. **Initialization**: Agents receive configuration and initial state
2. **Execution**: Each agent processes state independently
3. **State Transition**: Updated state passed to next agent
4. **Error Handling**: Agents can add errors to shared error list
5. **Completion**: Final state contains all agent outputs

## LLM Reasoning Implementation

### Structured Prompt Engineering

The system uses **structured prompts** for consistent LLM reasoning:

```python
def construct_analysis_prompt(feature_desc: str) -> str:
    """
    Engineered prompt for consistent LLM analysis with proper agent definition
    """
    return f"""You are an agent for feature analysis and factor assessment. Your role is to analyze product features and estimate normalized business factors.

From the feature description below, estimate normalized factors in [0,1] range:
- reach: How many users/customers will this impact? (0.0=very few, 1.0=almost all)
- revenue: How much will this contribute to revenue growth? (0.0=none, 1.0=significant)  
- risk_reduction: How much does this reduce business/technical risk? (0.0=none, 1.0=critical)
- engineering: How much engineering effort is required? (0.0=minimal, 1.0=massive)
- dependency: How many external dependencies/integrations? (0.0=none, 1.0=many complex)
- complexity: How complex is the implementation? (0.0=trivial, 1.0=extremely complex)

Feature Description: {feature_desc}

Return STRICT JSON with keys: reach, revenue, risk_reduction, engineering, dependency, complexity, notes (array of short analysis strings).

Respond with JSON in this exact format:
{{
  "reach": 0.5,
  "revenue": 0.5,
  "risk_reduction": 0.5,
  "engineering": 0.5,
  "dependency": 0.5,
  "complexity": 0.5,
  "notes": ["analysis point 1", "analysis point 2", "analysis point 3"]
}}"""
```

### Response Validation and Fallback

```python
def call_openai_json(model: str, prompt: str, **kwargs) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Robust LLM calling with validation
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        # Parse and validate JSON response
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        # Validate required fields
        required_fields = ['reach', 'revenue', 'risk_reduction', 'engineering']
        if all(field in parsed for field in required_fields):
            return parsed, None
        else:
            return None, "Missing required fields"
            
    except Exception as e:
        return None, str(e)
```

## Agent Intelligence Mechanisms

### 1. Heuristic Analysis (Fallback Mode)

```python
def _infer_defaults(description: str, config: Config) -> Dict[str, Any]:
    """
    Keyword-based heuristic analysis when LLM unavailable
    """
    defaults = initialize_defaults(config)
    desc_lower = description.lower()
    
    # Pattern matching with business rules
    for keyword, factors in config.keyword_mappings.items():
        if keyword in desc_lower:
            for factor, value in factors.items():
                # Use maximum value for factor enhancement
                if value > defaults[factor]:
                    defaults[factor] = value
                    defaults['notes'].append(f"High {factor} from '{keyword}'")
    
    return defaults
```

### 2. Contextual Scoring Logic

```python
def scorer_node(state: State, config: Config) -> State:
    """
    Agent applies configurable scoring algorithms
    """
    scored_features = []
    
    for feature in state.get('extracted', {}).get('features', []):
        # Calculate impact and effort based on policy
        if config.scoring_policy.name == "RICE":
            impact = calculate_rice_impact(feature)
            effort = calculate_rice_effort(feature)
            score = impact / max(effort, 0.1)  # Avoid division by zero
            
        elif config.scoring_policy.name == "ICE":
            impact = calculate_ice_impact(feature)
            effort = calculate_ice_effort(feature)
            score = impact / max(effort, 0.1)
            
        # Generate rationale (LLM-enhanced or template-based)
        rationale = _generate_rationale(feature, config) or f"Scored using {config.scoring_policy.name}"
        
        scored_features.append(ScoredFeature(
            name=feature.name,
            impact=impact,
            effort=effort, 
            score=score,
            rationale=rationale
        ))
    
    return {**state, 'scored': ScorerOutput(scored=scored_features)}
```

## Configuration and Extensibility

### Policy-Based Configuration

```python
class ScoringPolicy(BaseModel):
    """
    Configurable scoring algorithms
    """
    name: str
    impact_weights: Dict[str, float]  # Factor importance weights
    effort_weights: Dict[str, float]
    score_formula: str               # Mathematical formula
    
    @classmethod
    def rice_policy(cls) -> 'ScoringPolicy':
        return cls(
            name="RICE",
            impact_weights={
                "reach": 0.25,
                "revenue": 0.50, 
                "risk_reduction": 0.25
            },
            effort_weights={
                "engineering": 0.40,
                "complexity": 0.35,
                "dependency": 0.25
            },
            score_formula="impact / effort"
        )
```

### Agent Configuration Injection

```python
class Config(BaseModel):
    """
    Configuration dependency injection for agents
    """
    scoring_policy: ScoringPolicy
    llm_enabled: bool = False
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.3
    keyword_mappings: Dict[str, Dict[str, float]]
    
    @classmethod
    def rice_config(cls) -> 'Config':
        """Factory method for RICE configuration"""
        return cls(
            scoring_policy=ScoringPolicy.rice_policy(),
            keyword_mappings=get_business_keywords()
        )
```

## Error Handling and Resilience

### Graceful Degradation Strategy

```python
def robust_agent_execution(agent_func, state, config):
    """
    Wrapper for resilient agent execution
    """
    try:
        return agent_func(state, config)
    except LLMTimeoutError:
        # Fallback to heuristic analysis
        config_fallback = config.copy(update={"llm_enabled": False})
        return agent_func(state, config_fallback)
    except ValidationError as e:
        # Add error to state but continue processing
        errors = state.get('errors', [])
        errors.append(f"Validation error: {str(e)}")
        return {**state, 'errors': errors}
    except Exception as e:
        # Log error and return state with error notation
        errors = state.get('errors', [])
        errors.append(f"Agent error: {str(e)}")
        return {**state, 'errors': errors}
```

### Circuit Breaker for LLM Calls

```python
class LLMCircuitBreaker:
    """
    Circuit breaker pattern for LLM reliability
    """
    def __init__(self, failure_threshold=3, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = 0
        
    def call_with_breaker(self, llm_func, *args, **kwargs):
        if self.is_open():
            return None, "Circuit breaker open"
        
        try:
            result = llm_func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            return None, str(e)
    
    def is_open(self):
        if self.failure_count >= self.failure_threshold:
            return time.time() - self.last_failure_time < self.timeout
        return False
```

## Performance Optimizations

### Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_llm_analysis(feature_hash: str, model: str) -> Dict[str, Any]:
    """
    Cache LLM responses for identical feature descriptions
    """
    # Implementation details...
```

### Parallel Processing Potential

```python
async def parallel_agent_execution(features: List[RawFeature], config: Config):
    """
    Future enhancement: Parallel processing for independent features
    """
    tasks = [
        asyncio.create_task(process_single_feature(feature, config))
        for feature in features
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return aggregate_results(results)
```

## Testing and Validation

### Agent Unit Testing

```python
def test_extractor_agent():
    """
    Unit test for feature extraction agent
    """
    # Given
    raw_feature = RawFeature(
        name="Test Feature",
        description="Machine learning powered search functionality"
    )
    config = Config.default()
    state = {'raw': [raw_feature], 'errors': []}
    
    # When
    result = extractor_node(state, config)
    
    # Then
    assert 'extracted' in result
    extracted = result['extracted'].features[0]
    assert extracted.engineering > 0.7  # ML = high engineering effort
    assert "machine learning" in str(extracted.notes).lower()
```

### Integration Testing

```python
def test_full_agent_pipeline():
    """
    End-to-end test of agent orchestration
    """
    graph = FeaturePrioritizationGraph(Config.rice_config())
    features = load_test_features()
    
    results = graph.get_detailed_results(features)
    
    assert 'stages' in results
    assert len(results['stages']) == 3  # All agents executed
    assert results['summary']['highest_score'] > 0
```

## Monitoring and Observability

### Agent Performance Metrics

```python
class AgentMetrics:
    """
    Performance monitoring for agent execution
    """
    def __init__(self):
        self.execution_times = {}
        self.success_rates = {}
        self.llm_usage = {}
    
    def record_execution(self, agent_name: str, duration: float, success: bool):
        self.execution_times[agent_name] = self.execution_times.get(agent_name, [])
        self.execution_times[agent_name].append(duration)
        
        self.success_rates[agent_name] = self.success_rates.get(agent_name, {'success': 0, 'total': 0})
        self.success_rates[agent_name]['total'] += 1
        if success:
            self.success_rates[agent_name]['success'] += 1
```

### LLM Usage Tracking

```python
def track_llm_usage(model: str, prompt_tokens: int, completion_tokens: int, cost: float):
    """
    Track LLM usage for cost and performance optimization
    """
    metrics = {
        'timestamp': datetime.now(),
        'model': model,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
        'estimated_cost': cost
    }
    
    # Log to monitoring system
    logger.info("LLM Usage", extra=metrics)
```

## Security and Privacy

### Data Sanitization

```python
def sanitize_feature_for_llm(feature: RawFeature) -> str:
    """
    Remove sensitive information before LLM processing
    """
    description = feature.description
    
    # Remove patterns that might contain sensitive data
    patterns_to_remove = [
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP addresses
    ]
    
    for pattern in patterns_to_remove:
        description = re.sub(pattern, '[REDACTED]', description)
    
    return description
```

### Audit Logging

```python
def audit_log_agent_decision(agent_name: str, input_data: Any, output_data: Any, config: Config):
    """
    Comprehensive audit logging for compliance
    """
    audit_entry = {
        'timestamp': datetime.now().isoformat(),
        'agent': agent_name,
        'input_hash': hashlib.sha256(str(input_data).encode()).hexdigest(),
        'output_hash': hashlib.sha256(str(output_data).encode()).hexdigest(),
        'config_version': config.version,
        'llm_enabled': config.llm_enabled,
        'user_id': get_current_user_id(),
        'session_id': get_session_id()
    }
    
    audit_logger.info("Agent Decision", extra=audit_entry)
```

This technical deep dive provides the implementation details that support the agentic architecture described in the main TPM guide. The system combines modern AI capabilities with solid software engineering principles to create a robust, scalable, and intelligent prioritization platform.
