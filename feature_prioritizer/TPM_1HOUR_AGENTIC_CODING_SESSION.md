# TPM 1-Hour Agentic Programming Learning Session

## 🎯 **Session Overview for Technical Program Managers**

This 1-hour session is designed for TPMs with limited coding experience to understand and modify agentic programming concepts. You'll learn by implementing the new **FeasibilityAgent** for risk assessment in the Feature Prioritization System.

## 📋 **Session Structure**

### **10 minutes**: Understanding Agentic Architecture & Risk Assessment
### **35 minutes**: Hands-on Implementation (FeasibilityAgent)
### **15 minutes**: Testing and Validation

## 🆕 **What's New in This Session**

**FeasibilityAgent Implementation**: You'll add a new intelligent agent that:
- ✅ **Evaluates delivery risk** for each feature
- ✅ **Applies risk penalties** to priority scores  
- ✅ **Uses AI analysis** with deterministic fallback
- ✅ **Provides feasibility ratings** (High/Medium/Low)

---

## 🧠 **Part 1: Agentic Architecture & Risk Assessment (10 minutes)**

### What Makes Code "Agentic"?

**Traditional Code:**
```python
if "payment" in description:
    return 0.8  # Fixed rule
```

**Agentic Code:**
```python
class FeasibilityAgent:
    def assess_feature(self, feature):
        """I am an agent that evaluates delivery risk"""
        # Agent reasons about specific risks and complexity
        return self.analyze_with_llm(feature) or self.deterministic_analysis(feature)
```

**Key Differences:**
- ✅ **Agents have identity** and specific roles (FeasibilityAgent specializes in risk)
- ✅ **Agents make decisions** based on context (AI analysis + fallback)
- ✅ **Agents can reason** and explain their decisions (risk rationale)
- ✅ **Agents collaborate** by passing enriched data to scoring

### The FeasibilityAgent Architecture

**Agent Flow:**
```
Extract Features → FeasibilityAgent → Scorer → Prioritizer
                      ↓
               [Risk Assessment]
               - Feasibility: High/Medium/Low
               - Risk Factor: 0.0-1.0
               - Confidence: Safe/MediumRisk/HighRisk
```

**Why This Matters for TPMs:**
- 🎯 **Better Prioritization**: Risk-adjusted scores prevent overcommitting to risky features
- 🔍 **Transparent Decisions**: Each feature gets an explained risk assessment
- 🛡️ **Fallback Safety**: Works even when AI/LLM is unavailable
- 📊 **Business Intelligence**: Risk patterns help with capacity planning

---

## 🛠 **Part 2: Hands-On FeasibilityAgent Implementation (35 minutes)**

### **Exercise 1: Configure Risk Assessment Settings (10 minutes)**
*Perfect for TPMs - Configure business rules without complex programming*

**What You'll Learn:**
- How to encode risk assessment policies
- How to configure agent behavior
- Impact of risk penalties on prioritization

**Your Task:** Configure the FeasibilityAgent settings

**File to Edit:** `config.py` (Add after imports, around line 8)

**Code to Add:**
```python
class RiskPolicy(BaseModel):
    """Risk assessment configuration for feasibility validation."""
    risk_penalty: float = Field(default=0.5, ge=0, le=1, description="Risk penalty multiplier")
    enable_feasibility_agent: bool = Field(default=True, description="Enable feasibility agent")
    use_llm_analysis: bool = Field(default=True, description="Use LLM for risk analysis")
```

**Then update the Config class to include:**
```python
    # Change this line (around line 28):
    llm_enabled: bool = Field(default=True, description="Enable LLM enhancements...")  # Changed from False
    
    # Add this field after monitoring:
    risk: RiskPolicy = Field(default_factory=lambda: RiskPolicy(), description="Risk assessment configuration")
```

**Customization Options:**
```python
# Conservative (high risk penalty)
risk_penalty: float = 0.7  # 70% penalty for high-risk features

# Aggressive (low risk penalty) 
risk_penalty: float = 0.3  # 30% penalty for high-risk features

# Disable LLM, use only deterministic analysis
use_llm_analysis: bool = False
```

**Business Impact:** Risk penalty directly affects final scores - high-risk features get lower priority.

---

### **Exercise 2: Create the FeasibilityAgent (20 minutes)**
*Learn how to create a specialized AI agent*

**What You'll Learn:**
- How to create an agent with dual-mode analysis (AI + deterministic)
- How agents use domain expertise for risk assessment
- How to implement agent collaboration patterns

**Your Task:** Create the FeasibilityAgent

**File to Create:** `agents_feasibility.py`

**Core FeasibilityAgent Code:**
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
        """Use LLM to analyze feasibility and risk factors."""
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
            result, error = call_llm_json(prompt, model=self.config.llm_model)
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
        high_risk_keywords = ['ai', 'machine learning', 'blockchain', 'real-time', 'integration']
        medium_risk_keywords = ['database', 'authentication', 'payment', 'security']
        
        notes_text = ' '.join(spec.notes).lower()
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
            feasibility, confidence = "High", "Safe"
        elif final_risk <= 0.6:
            feasibility, confidence = "Medium", "MediumRisk"  
        else:
            feasibility, confidence = "Low", "HighRisk"
        
        return {
            "feasibility": feasibility,
            "risk_factor": final_risk,
            "delivery_confidence": confidence,
            "reasoning": f"Deterministic: complexity={complexity_risk:.2f}, keywords={keyword_risk:.2f}"
        }
    
    def assess_feature(self, spec: FeatureSpec) -> FeatureSpec:
        """Assess risk for a single feature - Creates new FeatureSpec with risk data."""
        print(f"🤖 Analyzing feasibility for: {spec.name}")
        analysis = self._analyze_with_llm(spec)
        
        # Create new FeatureSpec with risk assessment data
        new_spec = FeatureSpec(
            name=spec.name, reach=spec.reach, revenue=spec.revenue,
            risk_reduction=spec.risk_reduction, engineering=spec.engineering,
            dependency=spec.dependency, complexity=spec.complexity,
            notes=spec.notes + [f"Risk Assessment: {analysis.get('reasoning', 'No reasoning')}"],
            feasibility=analysis["feasibility"],
            risk_factor=analysis["risk_factor"], 
            delivery_confidence=analysis["delivery_confidence"]
        )
        
        print(f"📊 Assessed {spec.name}: feasibility={new_spec.feasibility}, risk={new_spec.risk_factor:.3f}")
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
                enriched_specs.append(self.assess_feature(spec))
            except Exception as e:
                print(f"❌ Error assessing {spec.name}: {str(e)}")
                errors.append(f"FeasibilityAgent error for {spec.name}: {str(e)}")
                # Add original spec with default risk values
                enriched_specs.append(spec)
        
        print(f"✅ FeasibilityAgent.enrich completed - processed {len(enriched_specs)} features")
        return enriched_specs
```

**Key Agent Concepts Demonstrated:**
- **Dual Intelligence**: AI analysis with deterministic fallback
- **Risk Specialization**: Domain expertise in feasibility assessment  
- **Agent Collaboration**: Enriches features for downstream agents
- **Error Handling**: Graceful degradation when AI fails

---

### **Exercise 3: Integrate Agent into Workflow (5 minutes)**
*Learn how agents collaborate in processing pipelines*

**What You'll Learn:**
- How to integrate new agents into existing workflows
- How agent data flows between processing stages
- How to add agent monitoring and debugging

**Your Task:** Add the FeasibilityAgent to the processing pipeline

**Files to Edit:** 
1. `models.py` - Add risk fields
2. `nodes.py` - Add feasibility node
3. `graph.py` - Integrate into workflow

**Quick Integration Steps:**

**Step A:** Add risk fields to `models.py` FeatureSpec class:
```python
# Add after the notes field:
feasibility: Optional[str] = Field(None, description="Implementation feasibility: Low|Medium|High")
risk_factor: Optional[float] = Field(0.4, ge=0, le=1, description="Risk factor (0=safe, 1=risky)")
delivery_confidence: Optional[str] = Field(None, description="Confidence: Safe|MediumRisk|HighRisk")
```

**Step B:** Add feasibility node to `nodes.py`:
```python
# Add import:
from agents_feasibility import FeasibilityAgent

# Add node function:
def feasibility_node(state: State, config: Optional[Config] = None) -> State:
    """Agent 2: Feasibility Agent - Assesses implementation risk and delivery confidence."""
    print("🚀 Feasibility Node: Starting risk assessment")
    
    cfg = config or Config.default()
    extracted_output = state.get("extracted")
    if not extracted_output or not hasattr(extracted_output, 'features'):
        state["errors"] = state.get("errors", []) + ["feasibility_node:no_extracted_features"]
        return state
    
    feasibility_agent = FeasibilityAgent(cfg)
    enriched_features = feasibility_agent.enrich(extracted_output.features, state.get("errors", []))
    extracted_output.features = enriched_features
    state["extracted"] = extracted_output
    
    print("✅ Feasibility Node: Completed risk assessment")
    return state
```

**Step C:** Update workflow in `graph.py`:
```python
# Update import:
from nodes import extractor_node, feasibility_node, scorer_node, prioritizer_node

# Update _build_graph method to add the new flow:
workflow.add_node("feasibility", feasibility_with_config)
workflow.add_edge("extract", "feasibility")      # Extract → Feasibility  
workflow.add_edge("feasibility", "score")        # Feasibility → Score
```

**Agent Collaboration Pattern:**
```
Raw Features → Extract Agent → FeasibilityAgent → Scorer Agent → Prioritizer Agent
                     ↓                ↓                ↓             ↓
               [Normalize]      [Risk Assessment]   [Risk Penalty]  [Final Ranking]
```

---

## 🧪 **Part 3: Testing and Validation (15 minutes)**

### **Test Your FeasibilityAgent Implementation**

**1. Verify Configuration (3 minutes)**
```bash
# Test that configuration loads correctly
python -c "
from config import Config
config = Config.default()
print(f'✅ LLM enabled: {config.llm_enabled}')
print(f'✅ Risk penalty: {config.risk.risk_penalty}')
print(f'✅ Feasibility agent: {config.risk.enable_feasibility_agent}')
"
```

**Expected Output:**
```
✅ LLM enabled: True
✅ Risk penalty: 0.5
✅ Feasibility agent: True
```

**2. Test FeasibilityAgent Risk Assessment (5 minutes)**
```bash
# Test the agent with a simple feature
python -c "
from agents_feasibility import FeasibilityAgent
from config import Config
from models import FeatureSpec

config = Config.default()
agent = FeasibilityAgent(config)
feature = FeatureSpec(
    name='AI-Powered Recommendations',
    reach=0.8, revenue=0.7, risk_reduction=0.3,
    engineering=0.8, dependency=0.6, complexity=0.9,
    notes=['machine learning', 'real-time processing']
)

result = agent.assess_feature(feature)
print(f'✅ Feature: {result.name}')
print(f'✅ Feasibility: {result.feasibility}')
print(f'✅ Risk Factor: {result.risk_factor:.3f}')
print(f'✅ Confidence: {result.delivery_confidence}')
"
```

**Expected Output:**
```
🤖 Analyzing feasibility for: AI-Powered Recommendations
🔧 Using deterministic analysis for AI-Powered Recommendations
✅ Feature: AI-Powered Recommendations
✅ Feasibility: Low
✅ Risk Factor: 0.867
✅ Confidence: HighRisk
```

**3. Test Complete Pipeline (7 minutes)**
```bash
# Test the full system with risk assessment
cd feature_prioritizer
python run.py --file samples/features.json --metric RICE --auto-save --verbose
```

**What to Look For:**
```
🚀 Feasibility Node: Starting risk assessment
🚀 FeasibilityAgent.enrich called with 15 features
🤖 Analyzing feasibility for: One-Click Checkout
📊 Assessed One-Click Checkout: feasibility=High, risk=0.300
✅ Feasibility Node: Completed risk assessment
   Scoring 1: One-Click Checkout - feasibility=High, risk=0.300
Results saved to: results/prioritization_rice_YYYYMMDD_HHMMSS.json
```

**4. Verify Risk Penalty Applied:**
```bash
# Check that high-risk features get lower scores
head -10 results/prioritization_rice_*.csv
```

**Expected:** Features with `HighRisk` confidence should have lower final scores than similar features with `Safe` confidence.

**5. Test Fallback Mode:**
```bash
# Test without LLM (deterministic only)
mv .env .env.backup 2>/dev/null || true
python run.py --json '[{"name": "Test Feature", "description": "A test feature with AI and machine learning"}]' --verbose
mv .env.backup .env 2>/dev/null || true
```

**Expected:** Should see "🔧 Using deterministic analysis" messages and still produce risk assessments.

---

## 🎓 **Learning Outcomes**

By the end of this session, you'll understand:

### **1. Agentic Architecture Concepts**
- ✅ **Agent Specialization**: FeasibilityAgent focuses solely on risk assessment
- ✅ **Agent Collaboration**: Agents work together in a processing pipeline
- ✅ **Agent Intelligence**: Dual-mode operation (AI + deterministic fallback)
- ✅ **Agent Transparency**: Each agent explains its decisions and reasoning

### **2. Risk Assessment Implementation**
- ✅ **Business Configuration**: How to set risk policies and penalties
- ✅ **Agent Development**: Creating specialized agents for domain expertise
- ✅ **Pipeline Integration**: Adding new agents to existing workflows
- ✅ **Error Handling**: Implementing graceful degradation and fallbacks

### **3. Production-Ready AI Systems**
- ✅ **Dual Intelligence**: AI enhancement with deterministic safety nets
- ✅ **Configuration Management**: Business rules separated from code logic
- ✅ **Monitoring Integration**: Agent activities tracked and audited
- ✅ **Data Flow Design**: How enriched data flows between processing stages

---

## 📈 **Next Steps for TPMs**

### **Immediate Actions (This Week)**
1. **Customize Risk Keywords**: Add domain-specific high-risk patterns to `_analyze_deterministic`
2. **Adjust Risk Penalties**: Tune `risk_penalty` based on your team's risk tolerance
3. **Test with Real Data**: Use your actual feature backlogs to validate risk assessments

### **Short-term Expansion (Next Month)**
1. **Custom Risk Agents**: Create industry-specific FeasibilityAgents (FinTech, Healthcare, etc.)
2. **Advanced Risk Factors**: Add regulatory compliance, team expertise, technology maturity
3. **Integration Planning**: Connect risk assessments with existing project management tools

### **Long-term Vision (Next Quarter)**
1. **Multi-Agent Risk System**: Different agents for technical, business, and regulatory risk
2. **Learning Risk Models**: Agents that improve based on actual delivery outcomes
3. **Organization-wide Risk Intelligence**: Standardized risk assessment across all teams

---

## 🔧 **Troubleshooting Guide**

### **Common Issues:**

**1. Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'agents_feasibility'
# Solution: Ensure the file is created in the correct directory
ls -la agents_feasibility.py  # Should exist in feature_prioritizer/
```

**2. Configuration Errors**
```bash
# Error: AttributeError: 'Config' object has no attribute 'risk'
# Solution: Ensure RiskPolicy is added to config.py and imported correctly
```

**3. Risk Fields Missing**
```bash
# Error: TypeError: __init__() missing required positional arguments
# Solution: Add risk fields to FeatureSpec model in models.py
```

**4. LLM Not Working**
```bash
# Test LLM connectivity
python -c "
import os
print('OpenAI Key:', 'Found' if os.getenv('OPENAI_API_AGENT_KEY') else 'Missing')
"
# Solution: Create .env file with OPENAI_API_AGENT_KEY=your-key-here
```

---

## 💡 **Pro Tips for TPMs**

### **1. Start with Deterministic Analysis**
- Implement and test the keyword-based risk detection first
- Add LLM analysis only after the base system works
- Use deterministic as a validation baseline for LLM results

### **2. Customize Risk Keywords for Your Domain**
```python
# Example for Healthcare
high_risk_keywords = ['hipaa', 'patient-data', 'clinical-integration', 'fda-approval']

# Example for FinTech  
high_risk_keywords = ['regulatory', 'compliance', 'fraud-detection', 'real-time-payments']

# Example for E-commerce
high_risk_keywords = ['payment-processing', 'inventory-sync', 'real-time-pricing']
```

### **3. Risk Penalty Calibration**
```python
# Conservative approach (enterprise/regulated industries)
risk_penalty: float = 0.8  # High penalty discourages risky features

# Balanced approach (most teams)
risk_penalty: float = 0.5  # Moderate penalty for balanced decisions  

# Aggressive approach (startups/innovation teams)
risk_penalty: float = 0.2  # Low penalty encourages innovation despite risk
```

### **4. Monitor Agent Performance**
- Check that risk assessments align with your intuition
- Track delivery outcomes vs. risk predictions
- Adjust keywords and penalties based on real experience

---

## 🚀 **Ready to Start?**

**Follow the consolidated implementation guide:**

The complete step-by-step implementation is available in:
📄 **TPM_Activity.md** - Consolidated implementation guide with all steps

**Quick Start Commands:**
```bash
git clone https://github.com/nmansur0ct/Agents.git
cd Agents/feature_prioritizer
git checkout v2
pip install -r requirements.txt

# Follow the TPM_Activity.md guide for complete implementation
# This 1-hour session gives you the foundational concepts
```

**Your agentic risk assessment system awaits!** 🤖⚡

---

## 📚 **Additional Resources**

### **Reference Documents:**
- 📄 **TPM_Activity.md** - Complete implementation guide (Steps 3-12)
- 📄 **TPM_Activity_Verbose.md** - Detailed technical documentation  
- 📄 **TPM_CHEAT_SHEET.md** - Quick reference for common operations
- 📄 **FEATURE_PRIORITY_ARCHITECTURE.md** - System architecture overview

### **Sample Data for Testing:**
- 📁 **samples/features.json** - Basic feature set
- 📁 **samples/healthcare_features.csv** - Healthcare-specific features
- 📁 **samples/fintech_features.csv** - FinTech-specific features
- 📁 **samples/manufacturing_features.csv** - Manufacturing features

### **Key Learning Path:**
1. **This 1-hour session** → Understand agentic concepts + basic implementation
2. **TPM_Activity.md** → Complete FeasibilityAgent implementation  
3. **Real data testing** → Validate with your actual feature backlogs
4. **Customization** → Adapt to your specific business domain and risk tolerance
