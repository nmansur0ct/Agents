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