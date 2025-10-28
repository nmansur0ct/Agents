#  **Complete Guide: Adding FeasibilityAgent for Risk Assessment**

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

# Install required Python dependencies
pip install -r requirements.txt
```

### **Working with Authentication:**
```bash
# Option 1: Use token in URL (for one-time operations)
git clone https://GIT_TOKEN@github.com/nmansur0ct/Agents.git

# Option 2: Set up authenticated remote (recommended)
git remote add github-auth https://GIT_TOKEN@github.com/nmansur0ct/Agents.git
git push github-auth v2  # Push changes using token
```

---

## ** What You're Adding**
A new **FeasibilityAgent** that evaluates delivery risk and applies risk penalties to feature scores.

**System Architecture:**
- **Before:** 3 agents (Extract → Score → Prioritize)  
- **After:** 4 agents (Extract → **Feasibility** → Score → Prioritize)

**Key Features:**
- ✅ **Dual-Mode Analysis**: LLM-enhanced + Deterministic fallback
- ✅ **Risk Assessment**: Feasibility, risk_factor, delivery_confidence fields
- ✅ **Configurable**: Enable/disable agent, customize risk penalties
- ✅ **Robust Error Handling**: Graceful fallback when LLM unavailable
- ✅ **Universal LLM Support**: OpenAI + Gemini (LLPM gateway)

---

## ** Step-by-Step Implementation Guide**

### ** Step 1: Configure Environment Variables**
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
# It includes LLM API keys and configuration settings for risk assessment.

# OpenAI Configuration (Option 1)
OPENAI_API_AGENT_KEY=sk-proj-xxxxxx
LLM_PROVIDER=openai

# Gemini Configuration (Option 2) - LLPM Gateway Support
LLPM_KEY=your_jwt_token_here
LLM_PROVIDER=gemini

# Universal LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
LLM_TIMEOUT=30
```

** Critical Configuration Fix:**
Replace `sk-proj-xxxxxx` with your actual API key OR use `LLPM_KEY` for Gemini support.

---

### ** Step 2: Fix Default Configuration** 
**File:** `config.py` - **CRITICAL FIX for Empty Fields Issue**

** Location:** Line ~15, in the `Config` class

**❌ WRONG (causes empty feasibility fields):**
```python
llm_enabled: bool = Field(default=False, description="Enable LLM enhancements...")
```

**✅ CORRECT (enables FeasibilityAgent by default):**
```python
llm_enabled: bool = Field(default=True, description="Enable LLM enhancements...")
```

**Action 1:** Add RiskPolicy class after imports (around line 8):
```python
class RiskPolicy(BaseModel):
    """Risk assessment configuration for feasibility validation."""
    risk_penalty: float = Field(default=0.5, ge=0, le=1, description="Risk penalty multiplier")
    enable_feasibility_agent: bool = Field(default=True, description="Enable feasibility agent")
    use_llm_analysis: bool = Field(default=True, description="Use LLM for risk analysis")
```

**Action 2:** Add to Config class after the monitoring field:
```python
    # Risk assessment configuration
    risk: RiskPolicy = Field(default_factory=lambda: RiskPolicy(), description="Risk assessment configuration")
```

---

### ** Step 3: Add Risk Fields to Data Models**
**File:** `models.py`

** Location 1:** FeatureSpec class, after the `notes` field:
```python
    # Risk assessment fields - CRITICAL for output population
    feasibility: Optional[str] = Field(None, description="Implementation feasibility: Low|Medium|High")
    risk_factor: Optional[float] = Field(0.4, ge=0, le=1, description="Risk factor (0=safe, 1=risky)")
    delivery_confidence: Optional[str] = Field(None, description="Confidence: Safe|MediumRisk|HighRisk")
```

** Location 2:** ScoredFeature class, after the `rationale` field:
```python
    # Risk assessment fields (carried from FeatureSpec) - CRITICAL for CSV output
    feasibility: Optional[str] = Field(None, description="Implementation feasibility: Low|Medium|High")
    risk_factor: Optional[float] = Field(None, ge=0, le=1, description="Risk factor (0=safe, 1=risky)")
    delivery_confidence: Optional[str] = Field(None, description="Confidence: Safe|MediumRisk|HighRisk")
```

---

### ** Step 4: Create Enhanced LLM Utility (Universal Support)**
**File:** `llm_utils.py` - **ENHANCED with Gemini Support**

** Location:** Add after existing functions or replace if exists:
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
            llmp_key = os.getenv("LLMP_KEY") or os.getenv("LLPM_KEY")
            if not llmp_key:
                return None, "LLPM_KEY not found for Gemini provider"
            
            headers = {
                "Authorization": f"Bearer {llmp_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gemini-1.5-flash",  # Use Gemini model
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "response_format": {"type": "json_object"}
            }
            
            # LLPM gateway endpoint
            response = requests.post(
                "https://api.llmp.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                return None, f"LLPM API error: {response.status_code} - {response.text}"
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content), None
            
        elif provider == "openai":
            # Use OpenAI directly
            api_key = os.getenv("OPENAI_API_AGENT_KEY")
            if not api_key:
                return None, "OPENAI_API_AGENT_KEY not found"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                return None, f"OpenAI API error: {response.status_code} - {response.text}"
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
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

---

### ** Step 5: Create FeasibilityAgent with Enhanced Error Handling**
**File:** Create `agents_feasibility.py`

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
            print(f" LLM disabled, using deterministic analysis for {spec.name}")
            return self._analyze_deterministic(spec)
        
        print(f"🤖 Analyzing feasibility for: {spec.name}")
        
        # Enhanced prompt with escaped braces for f-string safety
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
{{
  "feasibility": "High",
  "risk_factor": 0.3,
  "delivery_confidence": "Safe",
  "reasoning": "explanation here"
}}
        """
        
        try:
            # Use universal LLM function with auto-detection
            result, error = call_llm_json(
                prompt=prompt,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                provider="auto"  # Auto-detect OpenAI or Gemini
            )
            
            if error or not result:
                print(f"❌ LLM failed for {spec.name}, using deterministic: {error or 'Empty result'}")
                return self._analyze_deterministic(spec)
            
            print(f"✅ LLM analysis complete for {spec.name}: {result.get('feasibility', 'Unknown')}")
            return result
            
        except Exception as e:
            print(f"⚠️ LLM error for {spec.name}: {str(e)}")
            return self._analyze_deterministic(spec)
    
    def _analyze_deterministic(self, spec: FeatureSpec) -> Dict[str, Any]:
        """Enhanced deterministic analysis using keywords and complexity scores."""
        print(f" Using deterministic analysis for {spec.name}")
        
        text = f"{spec.name} {' '.join(spec.notes)}".lower()
        
        # Enhanced risk keyword analysis
        high_risk = ["migration", "refactor", "architecture", "microservice", "ai", "algorithm", 
                    "blockchain", "real-time", "machine learning", "neural", "deep learning"]
        medium_risk = ["integration", "api", "external", "security", "performance", "scalability", 
                      "framework", "database", "authentication"]
        low_risk = ["ui", "dashboard", "report", "button", "form", "display", "styling", 
                   "cosmetic", "text", "color", "layout"]
        
        high_count = sum(1 for word in high_risk if word in text)
        medium_count = sum(1 for word in medium_risk if word in text)
        low_count = sum(1 for word in low_risk if word in text)
        
        complexity = (spec.complexity + spec.engineering + spec.dependency) / 3
        
        print(f"    Complexity score: {complexity:.2f}")
        print(f"    Risk keywords: high={high_count}, medium={medium_count}, low={low_count}")
        
        # Enhanced decision logic
        if high_count > 0 or complexity > 0.7:
            risk_factor = min(0.8, 0.6 + complexity * 0.2)
            result = {"feasibility": "Low", "risk_factor": risk_factor, "delivery_confidence": "HighRisk", 
                     "reasoning": f"High complexity ({complexity:.2f}) or high-risk keywords detected"}
        elif medium_count > 0 or complexity > 0.5:
            risk_factor = min(0.6, 0.4 + complexity * 0.2)
            result = {"feasibility": "Medium", "risk_factor": risk_factor, "delivery_confidence": "MediumRisk", 
                     "reasoning": f"Medium complexity ({complexity:.2f}) or medium-risk indicators"}
        else:
            risk_factor = max(0.2, complexity)
            result = {"feasibility": "High", "risk_factor": risk_factor, "delivery_confidence": "Safe", 
                     "reasoning": f"Low complexity ({complexity:.2f}) and safe implementation pattern"}
        
        print(f"   ✅ Deterministic result: {result}")
        return result
    
    def assess_feature(self, spec: FeatureSpec) -> FeatureSpec:
        """Assess risk for a single feature - CRITICAL: Creates new FeatureSpec object."""
        print(f"   Before: feasibility={spec.feasibility}, risk={spec.risk_factor}, confidence={spec.delivery_confidence}")
        
        analysis = self._analyze_with_llm(spec)
        
        # Create a new FeatureSpec with updated risk assessment - CRITICAL FIX
        updated_notes = spec.notes.copy()
        if "reasoning" in analysis:
            updated_notes.append(f"Risk Assessment: {analysis['reasoning']}")
        
        # Create NEW FeatureSpec with risk assessment data
        new_spec = FeatureSpec(
            name=spec.name,
            reach=spec.reach,
            revenue=spec.revenue,
            risk_reduction=spec.risk_reduction,
            engineering=spec.engineering,
            dependency=spec.dependency,
            complexity=spec.complexity,
            notes=updated_notes,
            # CRITICAL: Set risk assessment fields
            feasibility=analysis["feasibility"],
            risk_factor=analysis["risk_factor"],
            delivery_confidence=analysis["delivery_confidence"]
        )
        
        print(f"   After: feasibility={new_spec.feasibility}, risk={new_spec.risk_factor}, confidence={new_spec.delivery_confidence}")
        return new_spec
    
    def enrich(self, specs: List[FeatureSpec], errors: List[str]) -> List[FeatureSpec]:
        """Enrich all features with AI-enhanced risk assessment."""
        print(f" FeasibilityAgent.enrich called with {len(specs)} features")
        print(f" Configuration: enable_feasibility_agent={self.config.risk.enable_feasibility_agent}")
        print(f" Configuration: llm_enabled={self.config.llm_enabled}")
        print(f" Configuration: use_llm_analysis={self.config.risk.use_llm_analysis}")
        
        if not self.config.risk.enable_feasibility_agent:
            print("⚠️ FeasibilityAgent disabled by configuration")
            return specs
        
        enriched_specs = []
        for i, spec in enumerate(specs):
            try:
                print(f"📋 Processing feature {i+1}/{len(specs)}: {spec.name}")
                enriched_spec = self.assess_feature(spec)
                enriched_specs.append(enriched_spec)
                
                if self.monitor:
                    self.monitor.audit_trail.log_event(
                        agent_name="feasibility_agent",
                        event_type="feature_assessed",
                        details={
                            "feature": spec.name,
                            "feasibility": enriched_spec.feasibility,
                            "risk_factor": enriched_spec.risk_factor,
                            "confidence": enriched_spec.delivery_confidence
                        }
                    )
            except Exception as e:
                print(f"❌ Error processing {spec.name}: {str(e)}")
                errors.append(f"feasibility_agent_error:{spec.name}:{str(e)}")
                enriched_specs.append(spec)
        
        print(f"✅ FeasibilityAgent.enrich completed - processed {len(enriched_specs)} features")
        return enriched_specs
```

---

### ** Step 6: Fix Graph Compilation Issue**
**File:** `graph.py`

** Location:** `_build_graph` method 


**Action 1:** Update import to include feasibility_node:
```python
from nodes import extractor_node, feasibility_node, scorer_node, prioritizer_node
```

**Action 2:** Add feasibility node to workflow in `_build_graph` method:
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

        return workflow.compile()  # CRITICAL: Must return compiled workflow
       

## **🧪 Testing & Validation**

### **Quick Verification Commands:**
```bash
# Navigate to project directory
cd /Users/n0m08hp/Agents/feature_prioritizer

# Test imports work correctly
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
```

### **Test with Sample Data:**
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
# 🤖 extractor_agent: ✅ Success Rate: 100.0%
# 🤖 feasibility_agent: ✅ Success Rate: 100.0%  ← NEW AGENT WITH LLM
# 🤖 scorer_agent: ✅ Success Rate: 100.0%
# 🤖 prioritizer_agent: ✅ Success Rate: 100.0%
#
# 📝 AUDIT EVENTS: 8
# ├─ llm_analysis: 3 events (LLM risk assessments)
# ├─ feature_assessed: 3 events
# └─ agent_execution: 4 events
```

### **Test with Industry-Specific Datasets:**
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

### **Expected Output Validation:**
Your tests should show:
1. ✅ **Configuration loaded**: `llm_enabled: True`, `enable_feasibility_agent: True`
2. ✅ **Risk fields populated**: Features have feasibility, risk_factor, delivery_confidence
3. ✅ **LLM or fallback working**: Either LLM analysis or deterministic analysis
4. ✅ **Pipeline completion**: All 4 agents (Extract → Feasibility → Score → Prioritize)
5. ✅ **CSV output complete**: No empty feasibility/delivery_confidence columns

---

## **⚙️ Configuration Options**

### **Risk Assessment Settings:**
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

# LLM Configuration (when enabled)
config.llm_enabled = True
config.llm_model = "gpt-4o-mini"  # or "gpt-3.5-turbo" for faster analysis
```

### **LLM Provider Configuration:**
```properties
# OpenAI Configuration
LLM_PROVIDER=openai
OPENAI_API_AGENT_KEY=sk-proj-your_key_here

# Gemini Configuration (via LLPM gateway)
LLM_PROVIDER=gemini
LLPM_KEY=your_jwt_token_here

# Auto-detection (recommended)
# Leave LLM_PROVIDER blank - system will auto-detect based on available keys
```

### **Performance Comparison:**

| Mode | Speed | Accuracy | Cost | Use Case |
|------|-------|----------|------|----------|
| **LLM (OpenAI)** | Medium | High | $$ | Complex features, detailed analysis |
| **LLM (Gemini)** | Fast | High | $ | Cost-effective LLM analysis |
| **Deterministic** | Fastest | Good | Free | Simple features, batch processing |

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

## ** Troubleshooting Guide**

### ** Issue 1: "'NoneType' object has no attribute 'invoke'"**
**Root Cause:** Missing return statement in `graph.py`

**Solution:** In `graph.py`, `_build_graph` method:
```python
# ❌ WRONG
def _build_graph(self) -> StateGraph:
    # ... build workflow ...
    workflow.compile()  # Missing return!

# ✅ CORRECT  
def _build_graph(self) -> StateGraph:
    # ... build workflow ...
    return workflow.compile()  # MUST return compiled workflow
```

### ** Issue 2: "Arguments missing for parameters feasibility, risk_factor, delivery_confidence"**
**Root Cause:** FeatureSpec constructor missing risk fields

**Solution:** In `nodes.py`, `_extractor_node_impl`, update FeatureSpec creation:
```python
feature_spec = FeatureSpec(
    name=raw_feature.name,
    # ... existing fields ...
    notes=final_notes,
    # ADD THESE CRITICAL FIELDS:
    feasibility=None,
    risk_factor=0.4,
    delivery_confidence=None
)
```

### ** Issue 3: "Feasibility and delivery_confidence fields are empty"**
**Root Cause:** `llm_enabled=False` in config prevents FeasibilityAgent execution

**CRITICAL FIX:** In `config.py`, change default:
```python
# ❌ WRONG (causes empty fields)
llm_enabled: bool = Field(default=False, ...)

# ✅ CORRECT (enables FeasibilityAgent)
llm_enabled: bool = Field(default=True, ...)
```

**Additional Checks:**
1. **Verify configuration**:
```python
python -c "from config import Config; c=Config.default(); print(f'llm_enabled: {c.llm_enabled}, feasibility_agent: {c.risk.enable_feasibility_agent}')"
```

2. **Check debug output** should show:
   - ` FeasibilityAgent.enrich called with X features`
   - `🤖 Analyzing feasibility for: [feature name]`
   - `✅ LLM analysis complete` or ` Using deterministic analysis`

### ** Issue 4: "Invalid format specifier" in FeasibilityAgent**
**Root Cause:** Single braces in f-string prompt causing format errors

**Solution:** In `agents_feasibility.py`, escape braces in prompt:
```python
# ❌ WRONG
prompt = f"""
{
  "feasibility": "High",
  "risk_factor": 0.3
}
"""

# ✅ CORRECT  
prompt = f"""
{{
  "feasibility": "High",
  "risk_factor": 0.3
}}
"""
```

### ** Issue 5: Import errors for FeasibilityAgent**
**Solution:** 
1. Ensure `agents_feasibility.py` exists in `feature_prioritizer/` directory
2. Check import in `nodes.py`: `from agents_feasibility import FeasibilityAgent`
3. Verify all dependencies: `from config import Config`, `from models import FeatureSpec`

### ** Issue 6: "No OpenAI API key found" or LLM not working**
**Solutions:**
1. **Check `.env` file** exists in `feature_prioritizer/` directory:
```properties
# OpenAI
OPENAI_API_AGENT_KEY=sk-proj-your_key_here

# OR Gemini via LLPM
LLPM_KEY=your_jwt_token_here
```

2. **Verify API key format**:
   - OpenAI: starts with `sk-proj-` or `sk-`
   - LLPM: JWT token format

3. **Test API connectivity**:
```bash
python -c "
import os
print('OpenAI Key:', 'Found' if os.getenv('OPENAI_API_AGENT_KEY') else 'Missing')
print('LLPM Key:', 'Found' if os.getenv('LLPM_KEY') else 'Missing')
"
```

### ** Issue 7: Risk fields missing from CSV output**
**Solution:** Ensure BOTH models include risk fields:

1. **FeatureSpec** (models.py):
```python
feasibility: Optional[str] = Field(None, ...)
risk_factor: Optional[float] = Field(0.4, ...)
delivery_confidence: Optional[str] = Field(None, ...)
```

2. **ScoredFeature** (models.py):
```python
feasibility: Optional[str] = Field(None, ...)
risk_factor: Optional[float] = Field(None, ...)
delivery_confidence: Optional[str] = Field(None, ...)
```

3. **Scorer node** (nodes.py) passes fields safely:
```python
feasibility=getattr(feature, 'feasibility', None),
risk_factor=getattr(feature, 'risk_factor', None),
delivery_confidence=getattr(feature, 'delivery_confidence', None)
```

### ** Issue 8: "cfg is not defined" in scorer node**
**Solution:** Use `config` instead of `cfg`:
```python
# ❌ WRONG
risk_multiplier = 1 - (cfg.risk.risk_penalty * feature.risk_factor)

# ✅ CORRECT
risk_multiplier = 1 - (config.risk.risk_penalty * feature.risk_factor)
```

### ** Debug Mode Testing:**
```bash
# Enable comprehensive debugging
python -c "
import sys
sys.path.append('.')
from agents_feasibility import FeasibilityAgent
from config import Config
from models import FeatureSpec

# Test configuration
config = Config.default()
print(f'✅ Configuration: llm_enabled={config.llm_enabled}')
print(f'✅ Configuration: enable_feasibility_agent={config.risk.enable_feasibility_agent}')

# Test agent
agent = FeasibilityAgent(config)
feature = FeatureSpec(
    name='Test AI Feature',
    description='Complex machine learning recommendation system',
    reach=5, revenue=4, risk_reduction=3,
    engineering=5, dependency=4, complexity=5,
    notes=['AI', 'machine learning', 'real-time processing']
)

print(f'\\n🧪 Testing with high-risk feature...')
result = agent.assess_feature(feature)
print(f'✅ Result: feasibility={result.feasibility}, risk={result.risk_factor:.3f}, confidence={result.delivery_confidence}')

if result.feasibility and result.delivery_confidence:
    print('🎉 SUCCESS: Risk assessment fields properly populated!')
else:
    print('❌ FAILED: Risk assessment fields are empty')
"
```

---

## **📚 Quick Reference Summary**

### **Files Modified:**
1. **`.env`** - API keys and LLM configuration
2. **`config.py`** - RiskPolicy class + llm_enabled=True default  
3. **`models.py`** - Risk fields in FeatureSpec + ScoredFeature
4. **`llm_utils.py`** - Universal LLM function (OpenAI + Gemini)
5. **`agents_feasibility.py`** - NEW: FeasibilityAgent with dual-mode analysis
6. **`nodes.py`** - feasibility_node + extractor updates + scorer risk penalty
7. **`graph.py`** - Add feasibility node + fix return statement

### **Key Configuration:**
```python
# Critical settings for working system
llm_enabled = True  # Must be True for FeasibilityAgent
risk.enable_feasibility_agent = True
risk.use_llm_analysis = True  # or False for deterministic only
risk.risk_penalty = 0.5  # Adjust risk impact
```

### **Success Indicators:**
- ✅ All imports successful
- ✅ Configuration shows `llm_enabled: True`
- ✅ Debug output shows feature processing
- ✅ Risk fields populated: feasibility, risk_factor, delivery_confidence
- ✅ CSV output includes complete risk assessment data
- ✅ 4-agent pipeline: Extract → Feasibility → Score → Prioritize

### **Testing Command:**
```bash
cd /Users/n0m08hp/Agents/feature_prioritizer
python run.py --file samples/features.json --llm --monitoring --detailed
```

**Expected Output:**
```
🤖 extractor_agent: ✅ Success Rate: 100.0%
🤖 feasibility_agent: ✅ Success Rate: 100.0%  ← NEW AGENT
🤖 scorer_agent: ✅ Success Rate: 100.0%
🤖 prioritizer_agent: ✅ Success Rate: 100.0%
```

This comprehensive guide provides everything needed for new developers to successfully implement the FeasibilityAgent with proper risk assessment capabilities!
        
        return workflow.compile()  # CRITICAL: Must return compiled workflow
```

---

### ** Step 7: Add Feasibility Node to Processing Pipeline**
**File:** `nodes.py`

**Action 1:** Add import after line 17:
```python
from agents_feasibility import FeasibilityAgent
```

**Action 2:** Add feasibility node function after `extractor_node`:
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
    for i, f in enumerate(extracted_output.features):
        print(f"   Feature {i+1}: {f.name} - feasibility={f.feasibility}, risk={f.risk_factor}")
    
    feasibility_agent = FeasibilityAgent(cfg)
    enriched_features = feasibility_agent.enrich(extracted_output.features, state["errors"])
    
    print(f"📋 Feasibility Node: Got {len(enriched_features)} enriched features")
    for i, f in enumerate(enriched_features):
        print(f"   Enriched {i+1}: {f.name} - feasibility={f.feasibility}, risk={f.risk_factor}")
    
    extracted_output.features = enriched_features
    state["extracted"] = extracted_output
    
    print("✅ Feasibility Node: Completed risk assessment")
    return state
```

**Action 3:** Update extractor node to include risk fields in FeatureSpec creation:
** Location:** In `_extractor_node_impl`, find the FeatureSpec constructor and update:
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

---

### ** Step 8: Apply Risk Penalty in Scorer + Safe Field Access**
**File:** `nodes.py`

** Location:** In `scorer_node`, find where `ScoredFeature` is created and replace with:
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
# 🤖 extractor_agent: ✅ Success Rate: 100.0%
# 🤖 feasibility_agent: ✅ Success Rate: 100.0%  ← NEW AGENT WITH LLM
# 🤖 scorer_agent: ✅ Success Rate: 100.0%
# 🤖 prioritizer_agent: ✅ Success Rate: 100.0%
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

## **⚙️ Configuration Options**

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

## ** Troubleshooting Common Issues**

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

### **Issue 7: Feasibility fields missing from final output**
**Solution:** Ensure both FeatureSpec AND ScoredFeature models include risk assessment fields as shown in Step 3, and that the scorer node passes these fields when creating ScoredFeature objects as shown in Step 6.

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

---

### **Issue 9: Feasibility and delivery_confidence fields are empty**
**Root Cause:** FeasibilityAgent not properly updating FeatureSpec objects with risk assessment data.

**CRITICAL FIX:** Update the default configuration in `config.py`:

```python
# LLM configuration - CHANGE THIS LINE
llm_enabled: bool = Field(default=True, description="Enable LLM enhancements...")  # Changed from False to True
```

**Additional Solutions:**

1. **Check LLM Configuration**: Verify `.env` file has correct API keys and enable LLM:
```properties
LLM_ENABLED=True
LLM_PROVIDER=gemini  # or openai
```

2. **Verify FeasibilityAgent.assess_feature method**: Should create new FeatureSpec with risk data:
```python
def assess_feature(self, spec: FeatureSpec) -> FeatureSpec:
    analysis = self._analyze_with_llm(spec)
    
    return FeatureSpec(
        name=spec.name,
        # ... copy all existing fields ...
        feasibility=analysis["feasibility"],
        risk_factor=analysis["risk_factor"],
        delivery_confidence=analysis["delivery_confidence"]
    )
```

3. **Check Debug Output**: Should see messages like:
   - ` FeasibilityAgent.enrich called with X features`
   - `🤖 Analyzing feasibility for: [feature name]`
   - `✅ LLM analysis complete for [feature]: High/Medium/Low`
   - ` Assessed [feature]: feasibility=High, risk=0.3, confidence=Safe`

4. **Enable Configuration**: Ensure risk assessment is enabled:
```python
config.risk.enable_feasibility_agent = True
config.risk.use_llm_analysis = True  # or False for deterministic
```

5. **Fix Scorer Node**: Ensure safe attribute access:
```python
feasibility=getattr(feature, 'feasibility', None),
risk_factor=getattr(feature, 'risk_factor', None),
delivery_confidence=getattr(feature, 'delivery_confidence', None)
```
