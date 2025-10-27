#  **TPM Activity: Adding FeasibilityAgent for Risk Assessment**

## ** Repository Setup**

### **Git Repository Information:**
- **Repository URL:** `https://github.com/nmansur0ct/Agents.git`
- **Branch:** `v2`
- **Access Token:** `GIT_TOKEN`

### **Initial Checkout Commands:**
```bash
# Clone the repository
git clone https://github.com/nmansur0ct/Agents.git
cd Agents

# Switch to v2 branch (contains latest features)
git checkout v2

# Navigate to feature prioritizer directory
cd feature_prioritizer

# Verify you have the latest code
git pull origin v2
```


### **Working with Authentication:**
```bash
# Option 1: Use token in URL (for one-time operations)
git clone https://GIT_TOKEN@github.com/nmansur0ct/Agents.git

# Option 2: Set up authenticated remote (recommended)
git remote add github-auth https://GIT_TOKEN@github.com/nmansur0ct/Agents.git
git push github-auth v2  # Push changes using token
```

### **Create LLM configuration in .envfile**
```bash
crate file .env under /feature_prioritizer
add 
# This .env file contains environment variables for the feature prioritizer application.
# It includes the OpenAI API key and other configuration settings for LLM integration.
# OpenAI API Key
OPENAI_API_AGENT_KEY=sk-proj-xxxxxx
# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
LLM_TIMEOUT=30
```

---

## **What You're Adding**
A new **FeasibilityAgent** that evaluates delivery risk and applies risk penalties to feature scores.

**Before:** 3 agents (Extract → Score → Prioritize)  
**After:** 4 agents (Extract → **Feasibility** → Score → Prioritize)

---

## **⚡ Quick Implementation Steps**

### **Step 1: Configure Environment Variables**
**File:** Create `.env` in the `feature_prioritizer` folder

**Action:** Create the environment file for LLM configuration:
```bash
# Navigate to feature_prioritizer directory
cd feature_prioritizer

# Create .env file
touch .env
```

**Content for `.env` file:**
```properties
# This .env file contains environment variables for the feature prioritizer application.
# It includes the OpenAI API key and other configuration settings for LLM integration.
# OpenAI API Key
OPENAI_API_AGENT_KEY=sk-proj-xxxxxx
# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
LLM_TIMEOUT=30
```

**Note:** Replace `sk-proj-xxxxxx` with your actual OpenAI API key to enable LLM-enhanced risk analysis.

---

### **Step 2: Add Risk Configuration**
**File:** `config.py`

**Action:** Add after line 8 (after imports):
```python
class RiskPolicy(BaseModel):
    """Risk assessment configuration for feasibility validation."""
    risk_penalty: float = Field(default=0.5, ge=0, le=1, description="Risk penalty multiplier")
    enable_feasibility_agent: bool = Field(default=True, description="Enable feasibility agent")
    use_llm_analysis: bool = Field(default=True, description="Use LLM for risk analysis")
```

**Then:** Add to Config class after the monitoring field:
```python
    # Risk assessment configuration
    risk: RiskPolicy = Field(default_factory=lambda: RiskPolicy(), description="Risk assessment configuration")
```

---

### **Step 3: Add Risk Fields to Features**
**File:** `models.py`

**Action:** Add to FeatureSpec class after the `notes` field:
```python
    # Risk assessment fields
    feasibility: Optional[str] = Field(None, description="Implementation feasibility: Low|Medium|High")
    risk_factor: Optional[float] = Field(0.4, ge=0, le=1, description="Risk factor (0=safe, 1=risky)")
    delivery_confidence: Optional[str] = Field(None, description="Confidence: Safe|MediumRisk|HighRisk")
```

**Note:** You'll also need to update the FeatureSpec constructor in `_extractor_node_impl` to include these fields:
```python
# In nodes.py, update FeatureSpec constructor:
feature_spec = FeatureSpec(
    name=raw_feature.name,
    reach=_norm_hint(raw_feature.reach_hint, defaults['reach']),
    revenue=_norm_hint(raw_feature.revenue_hint, defaults['revenue']),
    risk_reduction=_norm_hint(raw_feature.risk_reduction_hint, defaults['risk_reduction']),
    engineering=_norm_hint(raw_feature.engineering_hint, defaults['engineering']),
    dependency=_norm_hint(raw_feature.dependency_hint, defaults['dependency']),
    complexity=_norm_hint(raw_feature.complexity_hint, defaults['complexity']),
    notes=final_notes,
    # Risk assessment fields (will be populated by FeasibilityAgent)
    feasibility=None,
    risk_factor=0.4,  # Default risk factor
    delivery_confidence=None
)
```

---

### **Step 4: Create FeasibilityAgent with LLM**
**File:** Create `agents_feasibility.py`

```python
"""

FeasibilityAgent - AI-Enhanced Risk Assessment Module
"""
from typing import List, Dict, Any, Optional
from config import Config
from models import FeatureSpec
from llm_utils import call_openai_json
from monitoring import get_system_monitor

class FeasibilityAgent:
    def __init__(self, config: Config):
        self.config = config
        self.monitor = get_system_monitor() if config.monitoring.enabled else None
    
    def _analyze_with_llm(self, spec: FeatureSpec) -> Dict[str, Any]:
        """Use LLM to analyze feasibility and risk factors."""
        if not self.config.llm_enabled or not self.config.risk.use_llm_analysis:
            return self._analyze_deterministic(spec)
        
        prompt = f"""
        Analyze the implementation feasibility and delivery risk for this feature:
        
        Feature Name: {spec.name}
        Engineering Effort: {spec.engineering:.2f} (0=low, 1=high)
        Dependency Complexity: {spec.dependency:.2f} (0=low, 1=high)
        Implementation Complexity: {spec.complexity:.2f} (0=low, 1=high)
        Notes: {' | '.join(spec.notes)}
        
        Assess:
        1. Implementation feasibility (Low/Medium/High)
        2. Risk factor (0.0-1.0, where 0=safe, 1=very risky)  
        3. Delivery confidence (Safe/MediumRisk/HighRisk)
        
        Consider: technical complexity, external dependencies, unknowns, potential blockers.
        """
        
        schema = {
            "type": "object",
            "properties": {
                "feasibility": {"type": "string", "enum": ["Low", "Medium", "High"]},
                "risk_factor": {"type": "number", "minimum": 0, "maximum": 1},
                "delivery_confidence": {"type": "string", "enum": ["Safe", "MediumRisk", "HighRisk"]},
                "reasoning": {"type": "string"}
            },
            "required": ["feasibility", "risk_factor", "delivery_confidence", "reasoning"]
        }
        
        try:
            result = call_openai_json(
                prompt=prompt,
                schema=schema,
                model=self.config.llm_model,
                max_tokens=300,
                temperature=0.3
            )
            
            # Log LLM usage
            if self.monitor:
                self.monitor.audit_trail.log_event(
                    agent_name="feasibility_agent",
                    event_type="llm_analysis", 
                    details={"feature": spec.name, "reasoning": result.get("reasoning", "")[:100]}
                )
            
            return result
            
        except Exception as e:
            # Fallback to deterministic analysis
            if self.monitor:
                self.monitor.audit_trail.log_event(
                    agent_name="feasibility_agent",
                    event_type="llm_fallback",
                    details={"feature": spec.name, "error": str(e)}
                )
            return self._analyze_deterministic(spec)
    
    def _analyze_deterministic(self, spec: FeatureSpec) -> Dict[str, Any]:
        """Deterministic analysis using keywords and complexity scores."""
        text = f"{spec.name} {' '.join(spec.notes)}".lower()
        
        # Risk keyword analysis
        high_risk = ["migration", "refactor", "architecture", "microservice", "ai", "algorithm", "blockchain", "real-time"]
        medium_risk = ["integration", "api", "external", "security", "performance", "scalability", "framework"]
        low_risk = ["ui", "dashboard", "report", "button", "form", "display", "styling", "cosmetic"]
        
        high_count = sum(1 for word in high_risk if word in text)
        medium_count = sum(1 for word in medium_risk if word in text)
        low_count = sum(1 for word in low_risk if word in text)
        
        complexity = (spec.complexity + spec.engineering + spec.dependency) / 3
        
        # Determine risk level
        if high_count > 0 or complexity > 0.7:
            risk_factor = min(0.8, 0.6 + complexity * 0.2)
            return {"feasibility": "Low", "risk_factor": risk_factor, "delivery_confidence": "HighRisk", "reasoning": f"High complexity ({complexity:.2f}) or high-risk keywords detected"}
        elif medium_count > 0 or complexity > 0.5:
            risk_factor = min(0.6, 0.4 + complexity * 0.2)
            return {"feasibility": "Medium", "risk_factor": risk_factor, "delivery_confidence": "MediumRisk", "reasoning": f"Medium complexity ({complexity:.2f}) or medium-risk indicators"}
        else:
            risk_factor = max(0.2, complexity)
            return {"feasibility": "High", "risk_factor": risk_factor, "delivery_confidence": "Safe", "reasoning": f"Low complexity ({complexity:.2f}) and safe implementation pattern"}
    
    def assess_feature(self, spec: FeatureSpec) -> FeatureSpec:
        """Assess risk for a single feature using LLM or deterministic method."""
        analysis = self._analyze_with_llm(spec)
        
        spec.feasibility = analysis["feasibility"]
        spec.risk_factor = analysis["risk_factor"] 
        spec.delivery_confidence = analysis["delivery_confidence"]
        
        # Add reasoning to notes
        if "reasoning" in analysis:
            spec.notes.append(f"Risk Assessment: {analysis['reasoning']}")
        
        return spec
    
    def enrich(self, specs: List[FeatureSpec], errors: List[str]) -> List[FeatureSpec]:
        """Enrich all features with AI-enhanced risk assessment."""
        if not self.config.risk.enable_feasibility_agent:
            return specs
        
        enriched_specs = []
        for spec in specs:
            try:
                enriched_spec = self.assess_feature(spec)
                enriched_specs.append(enriched_spec)
                
                if self.monitor:
                    self.monitor.audit_trail.log_event(
                        agent_name="feasibility_agent",
                        event_type="feature_assessed",
                        details={
                            "feature": spec.name,
                            "feasibility": spec.feasibility,
                            "risk_factor": spec.risk_factor,
                            "confidence": spec.delivery_confidence
                        }
                    )
            except Exception as e:
                errors.append(f"feasibility_agent_error:{spec.name}:{str(e)}")
                enriched_specs.append(spec)
        
        return enriched_specs
```

---

### **Step 5: Add Feasibility Node**
**File:** `nodes.py`

**Action 1:** Add import after line 17:
```python
from agents_feasibility import FeasibilityAgent
```

**Action 2:** Add function after `extractor_node`:
```python
def feasibility_node(state: State, config: Optional[Config] = None) -> State:
    """
    Agent 2: Feasibility Agent
    Assesses implementation risk and delivery confidence for features.
    """
    
    # Apply monitoring if enabled
    if config and config.monitoring.enabled:
        monitor = get_system_monitor()
        decorator = monitor.get_monitoring_decorator("feasibility_agent", "assess_risk")
        return decorator(_feasibility_node_impl)(state, config)
    else:
        return _feasibility_node_impl(state, config)

def _feasibility_node_impl(state: State, config: Optional[Config] = None) -> State:
    """Implementation of feasibility node with monitoring support."""
    cfg = config or Config.default()
    
    if "errors" not in state:
        state["errors"] = []
    
    extracted_output = state.get("extracted")
    if not extracted_output or not hasattr(extracted_output, 'features'):
        state["errors"].append("feasibility_node:no_extracted_features")
        return state
    
    feasibility_agent = FeasibilityAgent(cfg)
    enriched_features = feasibility_agent.enrich(extracted_output.features, state["errors"])
    extracted_output.features = enriched_features
    state["extracted"] = extracted_output
    
    return state
```

---

### **Step 6: Apply Risk Penalty in Scorer**
**File:** `nodes.py`

**Action:** In `scorer_node`, find where `ScoredFeature` is created and replace with:
```python
        # Apply risk penalty if available
        final_score = score
        risk_note = ""
        
        if hasattr(feature, 'risk_factor') and feature.risk_factor is not None:
            risk_multiplier = 1 - (config.risk.risk_penalty * feature.risk_factor)
            final_score = score * risk_multiplier
            risk_note = f" | Risk-adjusted ({feature.delivery_confidence})"
        
        scored_features.append(ScoredFeature(
            name=feature.name,
            impact=impact,
            effort=effort, 
            score=final_score,
            rationale=rationale + risk_note
        ))
```

---

### **Step 7: Update Graph**
**File:** `graph.py`

**Action 1:** Update import:
```python
from nodes import extractor_node, feasibility_node, scorer_node, prioritizer_node
```

**Action 2:** Add to `_build_graph` method:
```python
        # Add feasibility wrapper
        def feasibility_with_config(state: State) -> State:
            return feasibility_node(state, self.config)
        
        # Add all nodes
        workflow.add_node("extract", extractor_with_config)
        workflow.add_node("feasibility", feasibility_with_config)  # NEW
        workflow.add_node("score", scorer_with_config)
        workflow.add_node("prioritize", prioritizer_with_config)
        
        # Update flow
        workflow.set_entry_point("extract")
        workflow.add_edge("extract", "feasibility")     # NEW
        workflow.add_edge("feasibility", "score")       # NEW
        workflow.add_edge("score", "prioritize")
        workflow.add_edge("prioritize", END)
```

---

## **🧪 Test Your Implementation**

```bash
# Test basic functionality with monitoring
python run.py --file samples/features.json --monitoring --performance-report

# Test with LLM-enhanced risk analysis (requires API key)
python run.py --file samples/features.json --llm --monitoring --performance-report

# Test deterministic mode only (no LLM)
python run.py --file samples/features.json --monitoring

# Test with detailed output to see risk adjustments
python run.py --file samples/features.json --llm --detailed --monitoring

# Save monitoring report to see LLM usage
python run.py --file samples/features.json --llm --monitoring --save-monitoring-report feasibility_llm_test.json

# Expected output shows feasibility_agent with LLM usage
#  extractor_agent: ✅ Success Rate: 100.0%
#  feasibility_agent: ✅ Success Rate: 100.0%  ← NEW AGENT WITH LLM
#  scorer_agent: ✅ Success Rate: 100.0%
#  prioritizer_agent: ✅ Success Rate: 100.0%
#
# 📝 AUDIT EVENTS: 8
# ├─ llm_analysis: 3 events (LLM risk assessments)
# ├─ feature_assessed: 3 events
# └─ agent_execution: 4 events
```

### ** Test with Different Industry Datasets**

The repository includes multiple sample datasets for comprehensive testing:

```bash
# Healthcare features (high compliance requirements)
python run.py --file samples/healthcare_features.csv --llm --monitoring

# FinTech features (security and fraud focus)  
python run.py --file samples/fintech_features.csv --llm --monitoring

# Manufacturing features (high complexity AI systems)
python run.py --file samples/manufacturing_features.csv --llm --monitoring

# Smart City features (large scale IoT systems)
python run.py --file samples/smart_city_features.csv --llm --monitoring

# EdTech features (education technology)
python run.py --file samples/edtech_features.csv --llm --monitoring

# HR features (workforce management)
python run.py --file samples/hr_features.csv --llm --monitoring
```

These datasets provide different risk profiles and complexity patterns to thoroughly test the FeasibilityAgent's risk assessment capabilities.

---

## ** Configuration Options**

```python
# Risk Assessment Settings
config.risk.risk_penalty = 0.3  # Light penalty (30%)
config.risk.risk_penalty = 0.7  # Heavy penalty (70%)

# Agent Control
config.risk.enable_feasibility_agent = False  # Disable agent entirely

# LLM vs Deterministic Analysis
config.risk.use_llm_analysis = True   # Use LLM for enhanced analysis
config.risk.use_llm_analysis = False  # Use only deterministic keywords

# LLM Configuration (when enabled)
config.llm_enabled = True
config.llm_model = "gpt-4o-mini"  # or "gpt-3.5-turbo" for faster analysis
```

### **LLM vs Deterministic Comparison:**

| Mode | Speed | Accuracy | Cost | Use Case |
|------|-------|----------|------|----------|
| **LLM** | Slower | Higher | $$ | Complex features, nuanced analysis |
| **Deterministic** | Faster | Good | Free | Simple features, batch processing |

### **LLM Analysis Benefits:**
- ✅ Contextual understanding of feature descriptions
- ✅ Nuanced risk assessment based on implementation details  
- ✅ Learns from complex technical patterns
- ✅ Provides detailed reasoning in audit logs

### **Deterministic Analysis Benefits:**
- ✅ Instant results, no API calls
- ✅ Consistent, predictable outputs
- ✅ No external dependencies
- ✅ Perfect for CI/CD pipelines

---

## **🔧 Troubleshooting Common Issues**

### **Issue 1: "monitor_agent_execution is not defined"**
**Solution:** Use the conditional monitoring pattern as shown in Step 5, not a decorator.

### **Issue 2: "Arguments missing for parameters feasibility, risk_factor, delivery_confidence"**
**Solution:** Update the FeatureSpec constructor in `_extractor_node_impl` as shown in Step 3.

### **Issue 3: "cfg is not defined"**
**Solution:** Use `config` instead of `cfg` in the scorer node risk penalty code.

### **Issue 4: Import errors for FeasibilityAgent**
**Solution:** Make sure to create the `agents_feasibility.py` file exactly as shown in Step 4.

### **Issue 5: LLM functionality not working**
**Solution:** Ensure the `.env` file is created in the feature_prioritizer directory with a valid OpenAI API key as shown in Step 1.

### **Issue 6: "No OpenAI API key found" error**
**Solution:** Check that:
- `.env` file exists in the feature_prioritizer directory
- `OPENAI_API_AGENT_KEY` is set with a valid API key (starts with `sk-proj-` or `sk-`)
- The API key has sufficient credits and permissions

### **Quick Verification:**
```bash
# Test imports work correctly
cd /Users/n0m08hp/Agents/feature_prioritizer
python -c "from config import Config; from models import FeatureSpec; from nodes import feasibility_node; print('✅ All imports successful')"
```

---

### **Sample LLM Output:**
```json
{
  "name": "AI-Powered Recommendations",
  "feasibility": "Low",
  "risk_factor": 0.75,
  "delivery_confidence": "HighRisk", 
  "notes": ["Risk Assessment: High complexity AI feature requiring ML expertise, data pipeline setup, and model training infrastructure - significant unknowns in performance and accuracy"]
}
```
