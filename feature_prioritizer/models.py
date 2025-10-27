"""
Data Models Module - Pydantic Schemas for Feature Prioritization
Implements the LLD specification for type-safe data contracts.
"""

from typing import List, Optional, TypedDict, TYPE_CHECKING, Any, Dict
from datetime import datetime

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

# Monitoring and Observability Models

class AgentExecutionRecord(BaseModel):
    """Record of agent execution for monitoring and analysis."""
    agent_name: str = Field(..., description="Name of the agent")
    action: str = Field(..., description="Action performed by the agent")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    duration: float = Field(..., ge=0, description="Execution duration in seconds")
    success: bool = Field(..., description="Whether execution was successful")
    input_size: int = Field(default=0, ge=0, description="Size of input data")
    output_size: int = Field(default=0, ge=0, description="Size of output data")
    llm_calls: int = Field(default=0, ge=0, description="Number of LLM API calls made")
    llm_tokens: int = Field(default=0, ge=0, description="Total tokens used in LLM calls")
    error_message: Optional[str] = Field(default=None, description="Error message if execution failed")
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1, description="Confidence in the result")

class SystemHealthMetrics(BaseModel):
    """System-wide health and performance metrics."""
    uptime_seconds: float = Field(..., ge=0, description="System uptime in seconds")
    total_executions: int = Field(..., ge=0, description="Total number of agent executions")
    overall_success_rate: float = Field(..., ge=0, le=1, description="Overall system success rate")
    executions_per_minute: float = Field(..., ge=0, description="Average executions per minute")
    total_llm_calls: int = Field(default=0, ge=0, description="Total LLM API calls")
    total_llm_tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    avg_tokens_per_execution: float = Field(default=0, ge=0, description="Average tokens per execution")

class AgentPerformanceMetrics(BaseModel):
    """Performance metrics for individual agents."""
    agent_name: str = Field(..., description="Name of the agent")
    avg_duration: float = Field(..., ge=0, description="Average execution duration")
    min_duration: float = Field(..., ge=0, description="Minimum execution duration")
    max_duration: float = Field(..., ge=0, description="Maximum execution duration")
    total_executions: int = Field(..., ge=0, description="Total number of executions")
    success_rate: float = Field(..., ge=0, le=1, description="Success rate for this agent")
    error_rate: float = Field(..., ge=0, le=1, description="Error rate for this agent")
    total_llm_calls: int = Field(default=0, ge=0, description="Total LLM calls by this agent")
    total_llm_tokens: int = Field(default=0, ge=0, description="Total tokens used by this agent")
    avg_tokens_per_call: float = Field(default=0, ge=0, description="Average tokens per LLM call")
    mean_time_between_failures: float = Field(default=float('inf'), ge=0, description="MTBF in seconds")

class AuditLogEntry(BaseModel):
    """Structured audit log entry for agent decisions."""
    timestamp: datetime = Field(default_factory=datetime.now, description="Entry timestamp")
    agent_name: str = Field(..., description="Name of the agent")
    action: str = Field(..., description="Action performed")
    input_hash: str = Field(..., description="Hash of input data for integrity")
    output_hash: str = Field(..., description="Hash of output data for integrity")
    config_snapshot: Dict[str, Any] = Field(..., description="Configuration at time of execution")
    decision_factors: Dict[str, Any] = Field(..., description="Key factors in agent decision")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in the decision")
    execution_time: float = Field(..., ge=0, description="Execution duration in seconds")
    success: bool = Field(..., description="Whether operation was successful")
    error_details: Optional[str] = Field(default=None, description="Error details if operation failed")

class MonitoringReport(BaseModel):
    """Comprehensive monitoring report."""
    report_timestamp: datetime = Field(default_factory=datetime.now, description="Report generation time")
    system_health: SystemHealthMetrics = Field(..., description="Overall system health metrics")
    agent_performance: Dict[str, AgentPerformanceMetrics] = Field(..., description="Per-agent performance metrics")
    audit_summary: Dict[str, Any] = Field(..., description="Audit trail summary")
    recommendations: List[str] = Field(default_factory=list, description="Performance improvement recommendations")

class AlertThresholds(BaseModel):
    """Configurable thresholds for monitoring alerts."""
    min_success_rate: float = Field(default=0.95, ge=0, le=1, description="Minimum acceptable success rate")
    max_avg_duration: float = Field(default=30.0, gt=0, description="Maximum acceptable average duration (seconds)")
    max_error_rate: float = Field(default=0.05, ge=0, le=1, description="Maximum acceptable error rate")
    min_confidence_score: float = Field(default=0.7, ge=0, le=1, description="Minimum acceptable confidence score")
    max_llm_tokens_per_execution: int = Field(default=10000, gt=0, description="Maximum tokens per execution")

class MonitoringConfig(BaseModel):
    """Configuration for monitoring and observability features."""
    enabled: bool = Field(default=True, description="Whether monitoring is enabled")
    log_directory: str = Field(default="logs", description="Directory for log files")
    audit_trail_enabled: bool = Field(default=True, description="Whether to maintain audit trail")
    performance_metrics_enabled: bool = Field(default=True, description="Whether to collect performance metrics")
    alert_thresholds: AlertThresholds = Field(default_factory=AlertThresholds, description="Alert thresholds")
    report_interval_minutes: int = Field(default=60, gt=0, description="Interval for generating monitoring reports")
    retention_days: int = Field(default=30, gt=0, description="Number of days to retain monitoring data")