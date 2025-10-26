"""
Data Models Module - Pydantic Schemas for Feature Prioritization
Implements the LLD specification for type-safe data contracts.
"""

from typing import List, Optional, TypedDict, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config
from pydantic import BaseModel, Field, validator

class RawFeature(BaseModel):
    """Input feature with optional numeric hints (1-5 scale)."""
    name: str = Field(..., description="Feature name")
    description: str = Field(..., description="Feature description")
    reach_hint: Optional[int] = Field(None, ge=1, le=5, description="Reach impact hint (1-5)")
    revenue_hint: Optional[int] = Field(None, ge=1, le=5, description="Revenue impact hint (1-5)")
    risk_reduction_hint: Optional[int] = Field(None, ge=1, le=5, description="Risk reduction hint (1-5)")
    engineering_hint: Optional[int] = Field(None, ge=1, le=5, description="Engineering effort hint (1-5)")
    dependency_hint: Optional[int] = Field(None, ge=1, le=5, description="Dependency complexity hint (1-5)")
    complexity_hint: Optional[int] = Field(None, ge=1, le=5, description="Implementation complexity hint (1-5)")

class FeatureSpec(BaseModel):
    """Normalized feature attributes (0-1 scale) with extraction notes."""
    name: str = Field(..., description="Feature name")
    reach: float = Field(..., ge=0, le=1, description="Normalized reach factor")
    revenue: float = Field(..., ge=0, le=1, description="Normalized revenue factor")
    risk_reduction: float = Field(..., ge=0, le=1, description="Normalized risk reduction factor")
    engineering: float = Field(..., ge=0, le=1, description="Normalized engineering effort")
    dependency: float = Field(..., ge=0, le=1, description="Normalized dependency complexity")
    complexity: float = Field(..., ge=0, le=1, description="Normalized implementation complexity")
    notes: List[str] = Field(default_factory=list, description="Extraction notes and reasoning")

class ScoredFeature(BaseModel):
    """Feature with computed impact, effort, and final score."""
    name: str = Field(..., description="Feature name")
    impact: float = Field(..., ge=0, le=1, description="Combined impact score")
    effort: float = Field(..., ge=0, le=1, description="Combined effort score (higher = more effort)")
    score: float = Field(..., description="Final prioritization score")
    rationale: str = Field(..., description="Scoring rationale and key factors")

class ExtractorOutput(BaseModel):
    """Output from the feature extraction agent."""
    features: List[FeatureSpec] = Field(..., description="List of normalized features")

class ScorerOutput(BaseModel):
    """Output from the scoring agent."""
    scored: List[ScoredFeature] = Field(..., description="List of scored features")

class PrioritizedOutput(BaseModel):
    """Final prioritized feature list."""
    ordered: List[ScoredFeature] = Field(..., description="Features ordered by priority score (desc)")

class State(TypedDict, total=False):
    """LangGraph state for orchestrating the prioritization pipeline."""
    raw: List[RawFeature]
    extracted: ExtractorOutput
    scored: ScorerOutput
    prioritized: PrioritizedOutput
    errors: List[str]

# Additional models for configuration and analysis
class ImpactWeights(BaseModel):
    """Weights for impact calculation."""
    reach: float = Field(0.4, ge=0, le=1, description="Weight for reach factor")
    revenue: float = Field(0.4, ge=0, le=1, description="Weight for revenue factor")
    risk_reduction: float = Field(0.2, ge=0, le=1, description="Weight for risk reduction factor")
    
    @validator('reach', 'revenue', 'risk_reduction')
    def validate_weights_sum(cls, v, values):
        """Validate that weights sum to approximately 1.0."""
        if len(values) == 2:  # All three values are set
            total = v + sum(values.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Impact weights must sum to 1.0, got {total}")
        return v

class EffortWeights(BaseModel):
    """Weights for effort calculation."""
    engineering: float = Field(0.5, ge=0, le=1, description="Weight for engineering effort")
    dependency: float = Field(0.3, ge=0, le=1, description="Weight for dependency complexity")
    complexity: float = Field(0.2, ge=0, le=1, description="Weight for implementation complexity")
    
    @validator('engineering', 'dependency', 'complexity')
    def validate_weights_sum(cls, v, values):
        """Validate that weights sum to approximately 1.0."""
        if len(values) == 2:  # All three values are set
            total = v + sum(values.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Effort weights must sum to 1.0, got {total}")
        return v