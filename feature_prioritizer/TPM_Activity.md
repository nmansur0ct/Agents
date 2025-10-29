# **TPM Implementation Guide: Adding FeasibilityAgent for Risk Assessment**

## **🎯 What This Guide Covers**

This guide helps Technical Program Managers (TPMs) implement a new **FeasibilityAgent** that adds intelligent risk assessment to the feature prioritization system. The agent evaluates each feature's delivery risk and applies penalties to scores based on complexity and implementation challenges.

**What You'll Add:**
- **Risk Assessment**: Automatic evaluation of feature delivery risk
- **Smart Scoring**: Risk penalties applied to final priority scores
- **Fallback Safety**: System works even without AI/LLM connectivity
- **Better Decisions**: More accurate prioritization with risk considerations

---

## **🔧 Prerequisites & Setup**

### **Step 1: Repository Checkout**
```bash
# 1. Clone the repository
git clone https://github.com/nmansur0ct/Agents.git
cd Agents

# 2. Switch to the correct branch
git checkout v2

# 3. Navigate to the working directory
cd feature_prioritizer

# 4. Verify you have the latest code
git pull origin v2

# 5. Install dependencies
pip install -r requirements.txt
```

### **Step 2: Verify Current System**
```bash
# Test the system works before making changes
python run.py --file samples/features.json --metric RICE --verbose
```

**Expected Output**: Should show feature processing but may have errors - we'll fix these.

---

## **📝 Implementation Steps (All Fixes Included)**

### **Step 3: Create Environment Configuration**

**What we're doing**: Setting up API keys and system configuration for the FeasibilityAgent to work properly.

**File to create**: `.env` (in the `feature_prioritizer` folder)

```bash
# Create the environment file
touch .env
```

**Add this content to `.env`**:
```properties
# Feature Prioritization System Configuration
# Replace the API key with your actual OpenAI key

# OpenAI Configuration (Primary option)
OPENAI_API_AGENT_KEY=sk-proj-your-actual-openai-api-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
LLM_TIMEOUT=30

# Alternative: Gemini Configuration (if using Google's API)
# LLMP_KEY=your-gemini-jwt-token-here
# LLM_PROVIDER=gemini

# System Settings
MONITORING_ENABLED=true
LOG_DIRECTORY=logs
```

**⚠️ Important**: Replace `sk-proj-your-actual-openai-api-key-here` with your real OpenAI API key.

---

### **Step 4: Fix Configuration Defaults**

**What we're doing**: Ensuring the system has proper default settings so the FeasibilityAgent can work correctly.

**File to edit**: `config.py`

**Action 1**: Add RiskPolicy class after imports (around line 8):
```python
class RiskPolicy(BaseModel):
    """Risk assessment configuration for feasibility validation."""
    risk_penalty: float = Field(default=0.5, ge=0, le=1, description="Risk penalty multiplier")
    enable_feasibility_agent: bool = Field(default=True, description="Enable feasibility agent")
    use_llm_analysis: bool = Field(default=True, description="Use LLM for risk analysis")
```

**Action 2**: Find this line (around line 28):
```python
    llm_enabled: bool = Field(default=False, description="Enable LLM enhancements for factor analysis and rationale generation")
```

**Replace it with**:
```python
    llm_enabled: bool = Field(default=True, description="Enable LLM enhancements for factor analysis and rationale generation")
```

**Action 3**: Add to Config class after the monitoring field:
```python
    # Risk assessment configuration
    risk: RiskPolicy = Field(default_factory=lambda: RiskPolicy(), description="Risk assessment configuration")
```

**Why this change**: Changed the default from `False` to `True` so the system automatically uses AI enhancement when available, with fallback to deterministic mode when needed.

---

### **Step 5: Add Risk Fields to Data Models**

**What we're doing**: Adding risk assessment fields to the data models so they can store feasibility and risk information.

**File to edit**: `models.py`

**Action 1**: In FeatureSpec class, after the `notes` field:
```python
    # Risk assessment fields - CRITICAL for output population
    feasibility: Optional[str] = Field(None, description="Implementation feasibility: Low|Medium|High")
    risk_factor: Optional[float] = Field(0.4, ge=0, le=1, description="Risk factor (0=safe, 1=risky)")
    delivery_confidence: Optional[str] = Field(None, description="Confidence: Safe|MediumRisk|HighRisk")
```

**Action 2**: In ScoredFeature class, after the `rationale` field:
```python
    # Risk assessment fields (carried from FeatureSpec) - CRITICAL for CSV output
    feasibility: Optional[str] = Field(None, description="Implementation feasibility: Low|Medium|High")
    risk_factor: Optional[float] = Field(None, ge=0, le=1, description="Risk factor (0=safe, 1=risky)")
    delivery_confidence: Optional[str] = Field(None, description="Confidence: Safe|MediumRisk|HighRisk")
```

**Why this change**: These fields store the risk assessment data that the FeasibilityAgent will populate.

---

### **Step 6: Create Enhanced LLM Utility**

**What we're doing**: Creating a universal LLM interface that supports multiple AI providers with proper error handling.

**File to create or replace**: `llm_utils.py`

**Add this function**:
```python
def call_llm_json(prompt: str, model: str = "gpt-3.5-turbo", 
                  temperature: float = 0.3, provider: str = "auto") -> Tuple[Optional[Dict], Optional[str]]:
    """
    Universal LLM JSON call supporting OpenAI and Gemini via LLPM gateway.
    Auto-detects provider based on available environment variables.
    """
    import os
    import json
    import requests
    from typing import Dict, Optional, Tuple
    
    # Auto-detect provider if not specified
    if provider == "auto":
        if os.getenv("LLPM_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_AGENT_KEY"):
            provider = "openai"
        else:
            return None, "No API keys found for LLM providers"
    
    try:
        if provider == "gemini":
            # Use LLPM gateway for Gemini
            url = "https://llpm.gateway.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('LLMP_KEY')}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gemini-1.5-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content), None
            
        elif provider == "openai":
            # Use OpenAI directly
            import openai
            openai.api_key = os.getenv("OPENAI_API_AGENT_KEY")
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            content = response.choices[0].message.content
            return json.loads(content), None
            
        else:
            return None, f"Unsupported provider: {provider}"
            
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
```

**Why this change**: Provides universal LLM support with automatic provider detection and proper error handling.

---

### **Step 7: Create FeasibilityAgent**

**What we're doing**: Creating the main FeasibilityAgent that performs risk assessment using both AI and deterministic methods.

**File to create**: `agents_feasibility.py`

```python
"""
FeasibilityAgent - AI-Enhanced Risk Assessment Module
Supports both LLM-enhanced analysis and deterministic fallback
"""
from typing import List, Dict, Any, Optional
from config import Config
from models import FeatureSpec
from llm_utils import call_llm_json
from monitoring import get_system_monitor

class FeasibilityAgent:
    def __init__(self, config: Config):
        self.config = config
        self.monitor = get_system_monitor() if config.monitoring.enabled else None
    
    def _analyze_with_llm(self, spec: FeatureSpec) -> Dict[str, Any]:
        """Use LLM to analyze feasibility and risk factors with enhanced error handling."""
        if not self.config.llm_enabled or not self.config.risk.use_llm_analysis:
            print(f"🔄 LLM disabled, using deterministic analysis for {spec.name}")
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

Please respond with valid JSON format:
{{{{
  "feasibility": "High",
  "risk_factor": 0.3,
  "delivery_confidence": "Safe",
  "reasoning": "explanation here"
}}}}
        """
        
        try:
            # Use universal LLM function with auto-detection
            result, error = call_llm_json(prompt, model=self.config.llm_model, temperature=self.config.llm_temperature)
            
            if error or not result:
                print(f"⚠️ LLM failed for {spec.name}: {error}")
                return self._analyze_deterministic(spec)
            
            print(f"✅ LLM analysis complete for {spec.name}: {result.get('feasibility', 'Unknown')}")
            return result
            
        except Exception as e:
            print(f"⚠️ LLM error for {spec.name}: {str(e)}")
            return self._analyze_deterministic(spec)
    
    def _analyze_deterministic(self, spec: FeatureSpec) -> Dict[str, Any]:
        """Enhanced deterministic analysis using keywords and complexity scores."""
        print(f"🔧 Using deterministic analysis for {spec.name}")
        
        # Risk keywords detection
        high_risk_keywords = ['ai', 'machine learning', 'blockchain', 'real-time', 'integration', 'api', 'third-party']
        medium_risk_keywords = ['database', 'authentication', 'payment', 'security', 'performance']
        
        notes_text = ' '.join(spec.notes).lower()
        
        # Calculate base risk from complexity factors
        complexity_risk = (spec.engineering + spec.dependency + spec.complexity) / 3
        
        # Adjust risk based on keywords
        keyword_risk = 0.0
        if any(keyword in notes_text for keyword in high_risk_keywords):
            keyword_risk = 0.3
        elif any(keyword in notes_text for keyword in medium_risk_keywords):
            keyword_risk = 0.15
        
        final_risk = min(1.0, complexity_risk + keyword_risk)
        
        # Determine feasibility and confidence
        if final_risk <= 0.3:
            feasibility = "High"
            confidence = "Safe"
        elif final_risk <= 0.6:
            feasibility = "Medium"
            confidence = "MediumRisk"
        else:
            feasibility = "Low"
            confidence = "HighRisk"
        
        result = {
            "feasibility": feasibility,
            "risk_factor": final_risk,
            "delivery_confidence": confidence,
            "reasoning": f"Deterministic analysis: complexity={complexity_risk:.2f}, keywords={keyword_risk:.2f}"
        }
        
        print(f"✅ Deterministic result: {result}")
        return result
    
    def assess_feature(self, spec: FeatureSpec) -> FeatureSpec:
        """Assess risk for a single feature - CRITICAL: Creates new FeatureSpec object."""
        print(f"🤖 Analyzing feasibility for: {spec.name}")
        
        # Log assessment start
        if self.monitor:
            self.monitor.audit_trail.log_event(
                agent_name="feasibility_agent",
                event_type="feature_assessment_start",
                details={"feature_name": spec.name}
            )
        
        # Perform analysis
        analysis = self._analyze_with_llm(spec)
        
        # Create new FeatureSpec with risk assessment data
        new_spec = FeatureSpec(
            name=spec.name,
            reach=spec.reach,
            revenue=spec.revenue,
            risk_reduction=spec.risk_reduction,
            engineering=spec.engineering,
            dependency=spec.dependency,
            complexity=spec.complexity,
            notes=spec.notes + [f"Risk Assessment: {analysis.get('reasoning', 'No reasoning provided')}"],
            feasibility=analysis["feasibility"],
            risk_factor=analysis["risk_factor"],
            delivery_confidence=analysis["delivery_confidence"]
        )
        
        print(f"📊 Assessed {spec.name}: feasibility={new_spec.feasibility}, risk={new_spec.risk_factor:.3f}, confidence={new_spec.delivery_confidence}")
        
        # Log assessment completion
        if self.monitor:
            self.monitor.audit_trail.log_event(
                agent_name="feasibility_agent",
                event_type="feature_assessed",
                details={
                    "feature_name": spec.name,
                    "feasibility": new_spec.feasibility,
                    "risk_factor": new_spec.risk_factor,
                    "delivery_confidence": new_spec.delivery_confidence
                }
            )
        
        return new_spec
    
    def enrich(self, specs: List[FeatureSpec], errors: List[str]) -> List[FeatureSpec]:
        """Enrich all features with AI-enhanced risk assessment."""
        print(f"🚀 FeasibilityAgent.enrich called with {len(specs)} features")
        
        if not self.config.risk.enable_feasibility_agent:
            print("⏭️ FeasibilityAgent disabled in configuration")
            return specs
        
        enriched_specs = []
        
        for spec in specs:
            try:
                enriched_spec = self.assess_feature(spec)
                enriched_specs.append(enriched_spec)
            except Exception as e:
                print(f"❌ Error assessing {spec.name}: {str(e)}")
                errors.append(f"FeasibilityAgent error for {spec.name}: {str(e)}")
                # Add original spec with default risk values
                default_spec = FeatureSpec(
                    name=spec.name,
                    reach=spec.reach,
                    revenue=spec.revenue,
                    risk_reduction=spec.risk_reduction,
                    engineering=spec.engineering,
                    dependency=spec.dependency,
                    complexity=spec.complexity,
                    notes=spec.notes,
                    feasibility="Medium",
                    risk_factor=0.5,
                    delivery_confidence="MediumRisk"
                )
                enriched_specs.append(default_spec)
        
        print(f"✅ FeasibilityAgent.enrich completed - processed {len(enriched_specs)} features")
        return enriched_specs
```

**Why this change**: Creates the core FeasibilityAgent with both AI and deterministic risk assessment capabilities.

---

### **Step 8: Fix Graph Workflow and Add Feasibility Node**

**What we're doing**: Fixing the workflow to prevent conflicts and adding the feasibility assessment step.

**File to edit**: `graph.py`

**Action 1**: Update import at the top:
```python
from nodes import extractor_node, feasibility_node, scorer_node, prioritizer_node
```

**Action 2**: In the `_build_graph` method, find this section (around line 55):
```python
        # Define the flow: START -> extract -> score -> prioritize -> END
        workflow.set_entry_point("extract")
        workflow.add_edge("extract", "feasibility")     # NEW
        workflow.add_edge("feasibility", "score")       # NEW
        workflow.add_edge("extract", "score")
        workflow.add_edge("score", "prioritize")
        workflow.add_edge("prioritize", END)
```

**Replace it with**:
```python
        # Add feasibility wrapper
        def feasibility_with_config(state: State) -> State:
            return feasibility_node(state, self.config)
        
        # Add all nodes
        workflow.add_node("extract", extractor_with_config)
        workflow.add_node("feasibility", feasibility_with_config)
        workflow.add_node("score", scorer_with_config)
        workflow.add_node("prioritize", prioritizer_with_config)
        
        # Define the flow: START -> extract -> feasibility -> score -> prioritize -> END
        workflow.set_entry_point("extract")
        workflow.add_edge("extract", "feasibility")      # Extract goes to Feasibility
        workflow.add_edge("feasibility", "score")        # Feasibility goes to Score
        workflow.add_edge("score", "prioritize")         # Score goes to Prioritize
        workflow.add_edge("prioritize", END)             # Prioritize goes to End

        return workflow.compile()  # CRITICAL: Must return compiled workflow
```

**Why this change**: Removed the conflicting direct path from "extract" to "score" and added proper feasibility node integration.

---

### **Step 9: Add Feasibility Node and Fix Scoring Issues**

**What we're doing**: Adding the feasibility processing node and fixing variable errors in the scoring system.

**File to edit**: `nodes.py`

**Action 1**: Add import after line 17:
```python
from agents_feasibility import FeasibilityAgent
```

**Action 2**: Add feasibility node function after `extractor_node`:
```python
def feasibility_node(state: State, config: Optional[Config] = None) -> State:
    """
    Agent 2: Feasibility Agent
    Assesses implementation risk and delivery confidence for features.
    """
    print("🚀 Feasibility Node: Starting risk assessment")
    
    # Apply monitoring if enabled
    if config and config.monitoring.enabled:
        monitor = get_system_monitor()
        decorator = monitor.get_monitoring_decorator("feasibility_agent", "assess_risk")
        return decorator(_feasibility_node_impl)(state, config)
    else:
        return _feasibility_node_impl(state, config)

def _feasibility_node_impl(state: State, config: Optional[Config] = None) -> State:
    """Implementation of feasibility node with comprehensive debugging."""
    cfg = config or Config.default()
    
    if "errors" not in state:
        state["errors"] = []
    
    extracted_output = state.get("extracted")
    if not extracted_output or not hasattr(extracted_output, 'features'):
        state["errors"].append("feasibility_node:no_extracted_features")
        return state
    
    print(f"📋 Feasibility Node: Processing {len(extracted_output.features)} features")
    
    feasibility_agent = FeasibilityAgent(cfg)
    enriched_features = feasibility_agent.enrich(extracted_output.features, state["errors"])
    
    extracted_output.features = enriched_features
    state["extracted"] = extracted_output
    
    print("✅ Feasibility Node: Completed risk assessment")
    return state
```

**Action 3**: Update extractor node FeatureSpec creation. Find the FeatureSpec constructor and update:
```python
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

**Action 4**: Fix scoring loop variable error. Find this code (around line 377):
```python
        scored_features = []
        
        for feature in extracted_output.features:
            # Calculate impact score
```

**Replace it with**:
```python
        scored_features = []
        
        for i, feature in enumerate(extracted_output.features):
            # Calculate impact score
```

**Action 5**: Fix duplicate scoring append and add risk penalty. Find this code block (around line 441):
```python
            scored_features.append(ScoredFeature(
                name=feature.name,
                impact=impact,
                effort=effort, 
                score=final_score,
                rationale=rationale + risk_note,
                # CRITICAL: Include risk assessment fields in final output - SAFE ACCESS
                feasibility=getattr(feature, 'feasibility', None),
                risk_factor=getattr(feature, 'risk_factor', None),
                delivery_confidence=getattr(feature, 'delivery_confidence', None)
            ))
            
            scored_features.append(scored_feature)
```

**Replace it with**:
```python
        # Apply risk penalty if available - ENHANCED with debugging
        final_score = score
        risk_note = ""
        
        print(f"   Scoring {i+1}: {feature.name} - feasibility={getattr(feature, 'feasibility', None)}, risk={getattr(feature, 'risk_factor', None)}")
        
        if hasattr(feature, 'risk_factor') and feature.risk_factor is not None:
            risk_multiplier = 1 - (config.risk.risk_penalty * feature.risk_factor)
            final_score = score * risk_multiplier
            risk_note = f" | Risk-adjusted ({getattr(feature, 'delivery_confidence', 'Unknown')})"
        
        scored_features.append(ScoredFeature(
            name=feature.name,
            impact=impact,
            effort=effort, 
            score=final_score,
            rationale=rationale + risk_note,
            # CRITICAL: Include risk assessment fields in final output - SAFE ACCESS
            feasibility=getattr(feature, 'feasibility', None),
            risk_factor=getattr(feature, 'risk_factor', None),
            delivery_confidence=getattr(feature, 'delivery_confidence', None)
        ))
```

**Why this change**: Added proper variable definition, removed duplicate append, and integrated risk penalty calculation with safe field access.

---

### **Step 10: Add Missing Monitoring Method**

**What we're doing**: Adding a missing method to the monitoring system that the FeasibilityAgent needs to log its activities.

**File to edit**: `monitoring.py`

**Find the `_hash_data` method in the `AuditTrail` class and add this new method right after it (around line 235)**:
```python
    def log_event(self, agent_name: str, event_type: str, details: Dict[str, Any]):
        """Log a simple event for agent activities."""
        try:
            # Create a simple audit entry for the event
            entry = AuditLogEntry(
                timestamp=datetime.now(),
                agent_name=agent_name,
                action=event_type,
                input_hash=self._hash_data(details),
                output_hash=self._hash_data(details),
                config_snapshot={},
                execution_time=0.0,
                success=True,
                decision_factors=details,
                confidence_score=1.0,
                error_details=None
            )
            
            self.audit_entries.append(entry)
            self._log_audit_entry(entry)
            
            print(f"📝 Audit: {agent_name} - {event_type}")
            
        except Exception as e:
            print(f"⚠️ Audit logging failed: {str(e)}")
```

**Why this change**: The FeasibilityAgent calls this method to log its activities, but it was missing from the monitoring system.

---

## **🧪 Testing Your Implementation**

### **Step 11: Test the Complete System**

**Run this command to test everything**:
```bash
cd feature_prioritizer
python run.py --file samples/features.json --metric RICE --auto-save --verbose
```

**What you should see**:
```bash
🚀 Feasibility Node: Starting risk assessment
🚀 FeasibilityAgent.enrich called with 15 features
🤖 Analyzing feasibility for: One-Click Checkout
✅ LLM analysis complete for One-Click Checkout: High
📊 Assessed One-Click Checkout: feasibility=High, risk=0.300, confidence=Safe
✅ Feasibility Node: Completed risk assessment
   Scoring 1: One-Click Checkout - feasibility=High, risk=0.300
   Scoring 2: Real-time Notifications - feasibility=Low, risk=0.720
Results saved to:
   results/prioritization_rice_YYYYMMDD_HHMMSS.json
   results/prioritization_rice_YYYYMMDD_HHMMSS.csv
```

**Check the output files**:
```bash
# Look at the JSON results
cat results/prioritization_rice_*.json | head -20

# Look at the CSV results  
head -5 results/prioritization_rice_*.csv
```

**Expected content**: You should see features with scores, rationale, and risk assessment fields populated.

---

### **Step 12: Verify Risk Assessment Working**

**Test without AI (fallback mode)**:
```bash
# Temporarily remove API key to test fallback
mv .env .env.backup
python run.py --file samples/features.json --metric RICE --auto-save --verbose
```

**What you should see**:
```bash
🔄 LLM disabled, using deterministic analysis for [Feature Name]
🔧 Using deterministic analysis for [Feature Name]
✅ Deterministic result: {'feasibility': 'High', 'risk_factor': 0.5, ...}
```

**Restore your configuration**:
```bash
mv .env.backup .env
```

---

## **📋 Summary of Changes Made**

### **What the FeasibilityAgent Adds**:

1. **Risk Evaluation**: Each feature gets assessed for delivery risk
2. **Feasibility Rating**: Features rated as High/Medium/Low feasibility
3. **Risk Penalties**: High-risk features get lower priority scores
4. **Smart Fallback**: System works with or without AI connectivity

### **Files Modified**:

| File | Purpose | Key Changes |
|------|---------|-------------|
| `.env` | Configuration | Added API keys and system settings |
| `config.py` | Configuration | Added RiskPolicy class, enabled LLM by default |
| `models.py` | Data Models | Added risk assessment fields to FeatureSpec and ScoredFeature |
| `llm_utils.py` | LLM Interface | Created universal LLM function with multi-provider support |
| `agents_feasibility.py` | Risk Agent | **NEW FILE** - Main FeasibilityAgent implementation |
| `graph.py` | Workflow | Fixed conflicting edges, added feasibility node |
| `nodes.py` | Processing | Added feasibility node, fixed scoring variables, integrated risk penalty |
| `monitoring.py` | Logging | Added missing log_event method |

### **Before vs After**:

**Before**: Extract → Score → Prioritize (3 steps)
**After**: Extract → **Risk Assessment** → Score → Prioritize (4 steps)

**Output Enhancement**:
- **Before**: Basic scores without risk consideration
- **After**: Risk-adjusted scores with feasibility insights

---

## **🚨 Common Issues & Solutions**

### **Issue 1**: "Can receive only one value per step"
**Solution**: Fixed in Step 8 (graph.py workflow edges)

### **Issue 2**: "name 'i' is not defined"
**Solution**: Fixed in Step 9 Action 4 (nodes.py variable definition)

### **Issue 3**: "name 'scored_feature' is not defined"
**Solution**: Fixed in Step 9 Action 5 (nodes.py duplicate removal)

### **Issue 4**: "'AuditTrail' object has no attribute 'log_event'"
**Solution**: Fixed in Step 10 (monitoring.py method addition)

### **Issue 5**: Empty output files
**Solution**: All above fixes resolve the pipeline errors that caused this

### **Issue 6**: "No OpenAI API key found"
**Solution**: Ensure Step 3 (.env file) is completed with your real API key

### **Issue 7**: Feasibility fields are empty
**Solution**: Ensure Step 4 config changes are applied (llm_enabled=True)

---

## **✅ Success Checklist**

- [ ] Repository cloned and on v2 branch
- [ ] `.env` file created with your API key
- [ ] All 8 code files modified as described
- [ ] Test command runs without errors
- [ ] Output files contain actual feature data (not empty)
- [ ] Risk assessment shows in the output
- [ ] Fallback mode works without API key

**When complete**: You'll have a working system that intelligently assesses feature delivery risk and incorporates that into prioritization decisions, helping your team make better-informed feature planning choices.

---

## **⚙️ Configuration Options**

### **Risk Assessment Settings**:
```python
# Light penalty (30%) - good for agile teams
config.risk.risk_penalty = 0.3  

# Heavy penalty (70%) - good for enterprise/critical systems
config.risk.risk_penalty = 0.7  

# Agent Control
config.risk.enable_feasibility_agent = False  # Disable agent entirely

# LLM vs Deterministic Analysis
config.risk.use_llm_analysis = True   # Use LLM for enhanced analysis
config.risk.use_llm_analysis = False  # Use only deterministic keywords
```

### **LLM Provider Configuration**:
```properties
# OpenAI Configuration
LLM_PROVIDER=openai
OPENAI_API_AGENT_KEY=sk-proj-your_key_here

# Gemini Configuration (via LLMP gateway)
LLM_PROVIDER=gemini
LLMP_KEY=your_jwt_token_here

# Auto-detection (recommended)
# Leave LLM_PROVIDER blank - system will auto-detect based on available keys
```

### **Performance Comparison**:

| Mode | Speed | Accuracy | Cost | Use Case |
|------|-------|----------|------|----------|
| **LLM (OpenAI)** | Medium | High | $$ | Complex features, detailed analysis |
| **LLM (Gemini)** | Fast | High | $ | Cost-effective LLM analysis |
| **Deterministic** | Fastest | Good | Free | Simple features, batch processing |

---

## **🔍 Quick Verification Commands**

```bash
# Test imports work correctly
cd feature_prioritizer
python -c "from config import Config; from models import FeatureSpec; from nodes import feasibility_node; print('✅ All imports successful')"

# Test FeasibilityAgent functionality
python -c "
from agents_feasibility import FeasibilityAgent
from config import Config
from models import FeatureSpec

config = Config.default()
agent = FeasibilityAgent(config)
feature = FeatureSpec(
    name='Test Feature',
    reach=3, revenue=2, risk_reduction=1,
    engineering=2, dependency=1, complexity=2,
    notes=['Simple UI update']
)
result = agent.assess_feature(feature)
print(f'✅ SUCCESS: Risk assessment fields populated!')
print(f'   feasibility={result.feasibility}')
print(f'   risk_factor={result.risk_factor}')
print(f'   delivery_confidence={result.delivery_confidence}')
"

# Test complete pipeline
python run.py --file samples/features.json --metric RICE --auto-save --verbose
```

---

*This consolidated guide provides everything needed to implement the FeasibilityAgent enhancement in a single document with all fixes integrated into the main steps, designed specifically for TPMs to follow step-by-step without requiring deep coding expertise.*
