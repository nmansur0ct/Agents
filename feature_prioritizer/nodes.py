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
        # Construct LLM prompt for factor analysis
        prompt = f"""Analyze this feature description and assess the following factors on a 0.0 to 1.0 scale:

Feature: {feature_desc}

For each factor, provide a score (0.0-1.0) and brief rationale:

1. REACH: How many users/customers will this impact? (0.0=very few, 1.0=almost all)
2. REVENUE: How much will this contribute to revenue growth? (0.0=none, 1.0=significant)
3. RISK_REDUCTION: How much does this reduce business/technical risk? (0.0=none, 1.0=critical)
4. ENGINEERING: How much engineering effort is required? (0.0=minimal, 1.0=massive)
5. DEPENDENCY: How many external dependencies/integrations? (0.0=none, 1.0=many complex)
6. COMPLEXITY: How complex is the implementation? (0.0=trivial, 1.0=extremely complex)

Respond with JSON in this exact format:
{{
  "reach": 0.5,
  "reach_rationale": "brief explanation",
  "revenue": 0.5,
  "revenue_rationale": "brief explanation", 
  "risk_reduction": 0.5,
  "risk_reduction_rationale": "brief explanation",
  "engineering": 0.5,
  "engineering_rationale": "brief explanation",
  "dependency": 0.5,
  "dependency_rationale": "brief explanation",
  "complexity": 0.5,
  "complexity_rationale": "brief explanation",
  "analysis_confidence": "high/medium/low",
  "key_insights": ["insight1", "insight2"]
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
            # Log key insights if available
            if 'key_insights' in response and response['key_insights']:
                for insight in response['key_insights'][:2]:  # Show first 2 insights
                    print(f"      💡 {insight}")
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
        # Construct LLM prompt for rationale generation
        prompt = f"""Generate a concise business rationale for this feature's priority score.

Feature: {feature.name}
Priority Score: {feature.score:.2f}

Factors:
- Impact Score: {feature.impact:.2f}
- Effort Score: {feature.effort:.2f}

Provide a 2-3 sentence business rationale explaining why this score makes sense and what the key trade-offs are. Focus on business value vs effort. Be specific and actionable.

Respond with JSON:
{{
  "rationale": "business rationale text here"
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
                        rationale_key = f"{factor}_rationale"
                        if rationale_key in llm_analysis:
                            final_notes.append(f"LLM {factor.replace('_', ' ')}: {llm_analysis[rationale_key]}")
                
                # Add overall insights
                if 'key_insights' in llm_analysis and llm_analysis['key_insights']:
                    final_notes.extend([f"LLM insight: {insight}" for insight in llm_analysis['key_insights']])
                
                if 'analysis_confidence' in llm_analysis:
                    final_notes.append(f"LLM confidence: {llm_analysis['analysis_confidence']}")
            
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