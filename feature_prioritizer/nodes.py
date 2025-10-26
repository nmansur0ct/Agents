"""
Agent Nodes Module - LangGraph Node Implementations
Implements the three core agents: Feature Extractor, Scorer, and Prioritizer.
"""

import re
import os
from typing import List, Dict, Any, Optional
from models import (
    RawFeature, FeatureSpec, ScoredFeature, 
    ExtractorOutput, ScorerOutput, PrioritizedOutput, 
    State
)
from config import Config, ScoringPolicy
from llm_utils import call_openai_json

def _norm_hint(hint_value: Optional[int] = None, default: float = 0.5) -> float:
    """
    Normalize hint value from 1-5 scale to 0-1 scale.
    
    Args:
        hint_value: User hint value (1-5), None if not provided
        default: Default value to use if hint is None
        
    Returns:
        Normalized value between 0 and 1
    """
    if hint_value is None:
        return default
    
    # Map 1-5 to 0-1 scale: 1->0, 2->0.25, 3->0.5, 4->0.75, 5->1.0
    return (hint_value - 1) / 4.0

def _infer_defaults(description: str, config: Config) -> Dict[str, Any]:
    """
    Infer default values from feature description using keyword heuristics.
    
    Args:
        description: Feature description text
        config: Configuration with keyword mappings
        
    Returns:
        Dictionary with inferred values and notes
    """
    # Initialize defaults
    defaults = {
        'reach': config.default_reach,
        'revenue': config.default_revenue,
        'risk_reduction': config.default_risk_reduction,
        'engineering': config.default_engineering,
        'dependency': config.default_dependency,
        'complexity': config.default_complexity,
        'notes': []
    }
    
    # Convert description to lowercase for matching
    desc_lower = description.lower()
    
    # Check for keyword matches
    for keyword, factors in config.keyword_mappings.items():
        if keyword in desc_lower:
            for factor, value in factors.items():
                if factor in defaults and factor != 'notes':
                    # Use the higher value between default and inferred
                    if value > defaults[factor]:
                        defaults[factor] = value
                        defaults['notes'].append(f"High {factor.replace('_', ' ')} inferred from '{keyword}'")
    
    # Add contextual notes based on combinations
    if defaults['engineering'] > 0.7 and defaults['complexity'] > 0.7:
        defaults['notes'].append("High complexity engineering effort")
    
    if defaults['revenue'] > 0.7 and defaults['reach'] > 0.7:
        defaults['notes'].append("High revenue potential with broad reach")
    
    if defaults['dependency'] > 0.6 and defaults['complexity'] > 0.6:
        defaults['notes'].append("Complex integration dependencies")
    
    return defaults

def _enhance_with_llm(feature_desc: str, config: Config) -> Dict[str, Any]:
    """
    Use LLM to analyze feature description and provide factor assessments.
    
    Args:
        feature_desc: Feature description text
        config: Configuration with LLM settings
        
    Returns:
        Dictionary with LLM-analyzed factors and rationale
    """
    if not config.llm_enabled:
        return {}
    
    print(f"🔍 LLM Factor Analysis requested for feature: '{feature_desc[:50]}{'...' if len(feature_desc) > 50 else ''}'")
    
    try:
        # Construct LLM prompt for factor analysis with anti-hallucination constraints
        prompt = f"""You are an agent for feature analysis and factor assessment. Your role is to analyze ONLY the provided feature description and estimate normalized business factors based on the given information.

CRITICAL INSTRUCTIONS:
- Base your analysis ONLY on the feature description provided
- Do NOT make assumptions about company size, industry, or user base unless explicitly stated
- Do NOT invent technical details not mentioned in the description
- Use conservative estimates when information is limited
- If a factor cannot be determined from the description, use 0.5 as a neutral estimate

Analyze this feature description and estimate factors in [0,1] range:

Feature Description: {feature_desc}

Factor Guidelines:
- reach: User impact based on description (0.0=very few users, 1.0=all users) 
- revenue: Revenue contribution based on stated benefits (0.0=no revenue impact, 1.0=major revenue driver)
- risk_reduction: Risk mitigation based on described problems (0.0=no risk reduction, 1.0=critical risk mitigation)
- engineering: Development effort based on technical complexity mentioned (0.0=trivial change, 1.0=major system rebuild)
- dependency: External integrations mentioned in description (0.0=no dependencies, 1.0=many complex integrations)
- complexity: Implementation difficulty based on feature scope (0.0=simple feature, 1.0=extremely complex system)

IMPORTANT: 
- Only reference information explicitly stated in the feature description
- Do not assume unstated technical requirements or business context
- Provide 2-3 brief analysis notes based only on the given description

Return STRICT JSON format:
{{
  "reach": 0.5,
  "revenue": 0.5,
  "risk_reduction": 0.5,
  "engineering": 0.5,
  "dependency": 0.5,
  "complexity": 0.5,
  "notes": ["observation based on description", "specific detail mentioned", "conservative assessment reason"]
}}"""

        # Call LLM
        response, error = call_openai_json(
            model=config.llm_model,
            api_key_env=config.llm_api_key_env,
            prompt=prompt,
            temperature=config.llm_temperature
        )
        
        if response and not error:
            print(f"   ✨ LLM Factor Analysis completed successfully")
            # Log analysis notes if available
            if 'notes' in response and response['notes']:
                for note in response['notes'][:3]:  # Show first 3 notes
                    print(f"      💡 {note}")
        else:
            print(f"   ⚠️  LLM Factor Analysis failed, falling back to keyword analysis")
        
        return response if response and not error else {}
        
    except Exception as e:
        print(f"   ❌ LLM Factor Analysis error: {str(e)}")
        # Silently fail and return empty dict - LLM is enhancement, not requirement
        return {}

def _generate_rationale(feature: ScoredFeature, config: Config) -> str:
    """
    Use LLM to generate business rationale for prioritization score.
    
    Args:
        feature: Scored feature with all factors
        config: Configuration with LLM settings
        
    Returns:
        Generated business rationale or empty string if LLM unavailable
    """
    if not config.llm_enabled:
        return ""
    
    print(f" LLM Rationale Generation requested for feature: '{feature.name}'")
    
    try:
        # Construct LLM prompt for rationale generation with anti-hallucination constraints
        prompt = f"""You are an agent for business rationale generation. Your role is to create factual, evidence-based business justifications using ONLY the provided data.

CRITICAL INSTRUCTIONS:
- Base your rationale ONLY on the provided scores and feature name
- Do NOT make assumptions about market conditions, competitors, or business strategy
- Do NOT invent technical details or implementation specifics
- Use only the numerical scores provided to justify the prioritization
- Keep explanations factual and conservative

Provided Data:
Feature Name: {feature.name}
Priority Score: {feature.score:.2f}
Impact Score: {feature.impact:.2f} 
Effort Score: {feature.effort:.2f}

Generate a 2-3 sentence business rationale that:
1. References the actual numerical scores provided
2. Explains the mathematical relationship between impact and effort
3. States the relative priority position without speculating on broader business context

IMPORTANT:
- Only use the feature name and scores provided
- Do not assume business domain, company size, or market conditions
- Do not invent strategic implications beyond what the scores indicate
- Keep language factual and evidence-based

Return STRICT JSON format:
{{
  "rationale": "factual rationale based only on provided scores and feature name"
}}"""

        # Call LLM
        response, error = call_openai_json(
            model=config.llm_model,
            api_key_env=config.llm_api_key_env,
            prompt=prompt,
            temperature=config.llm_temperature
        )
        
        if response and not error and 'rationale' in response:
            print(f"   ✨ LLM Rationale generated successfully")
            return response.get('rationale', '')
        else:
            print(f"   ⚠️  LLM Rationale generation failed, using default rationale")
            return ''
        
    except Exception as e:
        print(f"   ❌ LLM Rationale generation error: {str(e)}")
        # Silently fail and return empty string
        return ""

def extractor_node(state: State, config: Optional[Config] = None) -> State:
    """
    Agent 1: Feature Extractor
    Converts RawFeature list to normalized FeatureSpec list.
    """
    try:
        if config is None:
            config = Config.default()
        extracted_features = []
        
        for raw_feature in state.get('raw', []):
            # Get defaults from description analysis
            defaults = _infer_defaults(raw_feature.description, config)
            
            # Enhance with LLM analysis if enabled
            llm_analysis = _enhance_with_llm(raw_feature.description, config)
            
            # Merge LLM insights with defaults (LLM takes precedence if available)
            final_notes = defaults['notes'].copy()
            
            if llm_analysis:
                # Override defaults with LLM analysis where available
                for factor in ['reach', 'revenue', 'risk_reduction', 'engineering', 'dependency', 'complexity']:
                    if factor in llm_analysis and llm_analysis[factor] is not None:
                        defaults[factor] = max(0.0, min(1.0, llm_analysis[factor]))  # Clamp to valid range
                
                # Add LLM analysis notes
                if 'notes' in llm_analysis and llm_analysis['notes']:
                    final_notes.extend([f"LLM: {note}" for note in llm_analysis['notes']])
                
                final_notes.append(f"LLM factor analysis applied")
            
            # Build FeatureSpec with normalized values
            feature_spec = FeatureSpec(
                name=raw_feature.name,
                reach=_norm_hint(raw_feature.reach_hint, defaults['reach']),
                revenue=_norm_hint(raw_feature.revenue_hint, defaults['revenue']),
                risk_reduction=_norm_hint(raw_feature.risk_reduction_hint, defaults['risk_reduction']),
                engineering=_norm_hint(raw_feature.engineering_hint, defaults['engineering']),
                dependency=_norm_hint(raw_feature.dependency_hint, defaults['dependency']),
                complexity=_norm_hint(raw_feature.complexity_hint, defaults['complexity']),
                notes=final_notes
            )
            
            extracted_features.append(feature_spec)
        
        # Update state with extracted features
        state['extracted'] = ExtractorOutput(features=extracted_features)
        
    except Exception as e:
        errors = state.get('errors', [])
        errors.append(f"Extraction error: {str(e)}")
        state['errors'] = errors
    
    return state

def scorer_node(state: State, config: Optional[Config] = None) -> State:
    """
    Agent 2: Impact-Effort Scorer
    Computes impact and effort scores using weighted sums.
    """
    try:
        if config is None:
            config = Config.default()
        extracted_output = state.get('extracted')
        
        if not extracted_output or not extracted_output.features:
            errors = state.get('errors', [])
            errors.append("No extracted features found for scoring")
            state['errors'] = errors
            return state
        
        scored_features = []
        
        for feature in extracted_output.features:
            # Calculate impact score
            impact = (
                feature.reach * config.impact_weights.reach +
                feature.revenue * config.impact_weights.revenue +
                feature.risk_reduction * config.impact_weights.risk_reduction
            )
            
            # Calculate effort score
            effort = (
                feature.engineering * config.effort_weights.engineering +
                feature.dependency * config.effort_weights.dependency +
                feature.complexity * config.effort_weights.complexity
            )
            
            # Calculate final score based on policy
            if config.prioritize_metric == ScoringPolicy.RICE:
                # RICE proxy: (impact * reach_factor) / max(0.05, effort)
                reach_factor = 0.5 + 0.5 * feature.reach
                score = (impact * reach_factor) / max(0.05, effort)
                
            elif config.prioritize_metric == ScoringPolicy.ICE:
                # ICE: impact * (1 - effort)
                score = impact * (1 - effort)
                
            else:  # ImpactMinusEffort
                # Simple difference
                score = impact - effort
            
            # Generate rationale
            rationale_parts = []
            
            # High impact factors
            if feature.reach > 0.7:
                rationale_parts.append("High reach")
            if feature.revenue > 0.7:
                rationale_parts.append("Revenue impact")
            if feature.risk_reduction > 0.7:
                rationale_parts.append("Risk reduction")
            
            # High effort factors
            if feature.engineering > 0.7:
                rationale_parts.append("Engineering heavy")
            if feature.dependency > 0.7:
                rationale_parts.append("Complex dependencies")
            if feature.complexity > 0.7:
                rationale_parts.append("High complexity")
            
            # Add policy-specific rationale
            rationale_parts.append(f"Scored using {config.prioritize_metric.value}")
            
            rationale = "; ".join(rationale_parts) if rationale_parts else "Standard scoring"
            
            scored_feature = ScoredFeature(
                name=feature.name,
                impact=round(impact, 3),
                effort=round(effort, 3),
                score=round(score, 3),
                rationale=rationale
            )
            
            scored_features.append(scored_feature)
        
        # Update state with scored features
        state['scored'] = ScorerOutput(scored=scored_features)
        
    except Exception as e:
        errors = state.get('errors', [])
        errors.append(f"Scoring error: {str(e)}")
        state['errors'] = errors
    
    return state

def prioritizer_node(state: State, config: Optional[Config] = None) -> State:
    """
    Agent 3: Prioritizer
    Sorts features by score in descending order.
    """
    try:
        if config is None:
            config = Config.default()
            
        scored_output = state.get('scored')
        
        if not scored_output or not scored_output.scored:
            errors = state.get('errors', [])
            errors.append("No scored features found for prioritization")
            state['errors'] = errors
            # Create empty prioritized output
            state['prioritized'] = PrioritizedOutput(ordered=[])
            return state
        
        # Enhance rationales with LLM if enabled
        enhanced_features = []
        for feature in scored_output.scored:
            # Generate enhanced rationale using LLM
            llm_rationale = _generate_rationale(feature, config)
            
            # Combine existing rationale with LLM enhancement
            if llm_rationale:
                enhanced_rationale = f"{feature.rationale}\n\nLLM Analysis: {llm_rationale}"
            else:
                enhanced_rationale = feature.rationale
                
            # Create new feature with enhanced rationale
            enhanced_feature = ScoredFeature(
                name=feature.name,
                impact=feature.impact,
                effort=feature.effort,
                score=feature.score,
                rationale=enhanced_rationale
            )
            enhanced_features.append(enhanced_feature)
        
        # Sort features by score (descending)
        ordered_features = sorted(
            enhanced_features,
            key=lambda f: f.score,
            reverse=True
        )
        
        # Update state with prioritized features
        state['prioritized'] = PrioritizedOutput(ordered=ordered_features)
        
    except Exception as e:
        errors = state.get('errors', [])
        errors.append(f"Prioritization error: {str(e)}")
        state['errors'] = errors
    
    return state