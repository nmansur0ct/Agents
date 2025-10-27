"""
Configuration Module - Scoring Policies and Weights
Implements configurable prioritization strategies per LLD specification.
"""

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from models import ImpactWeights, EffortWeights, MonitoringConfig

class ScoringPolicy(str, Enum):
    """Available scoring policies for feature prioritization."""
    RICE = "RICE"
    ICE = "ICE"
    IMPACT_MINUS_EFFORT = "ImpactMinusEffort"

class Config(BaseModel):
    """Main configuration for feature prioritization."""
    
    # Scoring weights
    impact_weights: ImpactWeights = Field(default_factory=lambda: ImpactWeights(), description="Weights for impact calculation")
    effort_weights: EffortWeights = Field(default_factory=lambda: EffortWeights(), description="Weights for effort calculation")
    
    # Scoring policy
    prioritize_metric: ScoringPolicy = Field(default=ScoringPolicy.RICE, description="Prioritization scoring policy")
    
    # LLM configuration
    llm_enabled: bool = Field(default=False, description="Enable LLM enhancements for factor analysis and rationale generation")
    llm_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use for LLM calls")
    llm_api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable name for OpenAI API key")
    llm_temperature: float = Field(default=0.3, ge=0, le=2, description="Temperature for LLM calls (0=deterministic, 1=balanced, 2=creative)")
    llm_max_tokens: int = Field(default=500, ge=50, le=4000, description="Maximum tokens for LLM responses")
    llm_timeout: int = Field(default=30, ge=5, le=120, description="Timeout in seconds for LLM API calls")
    
    # Default values for missing hints
    default_reach: float = Field(default=0.5, ge=0, le=1, description="Default reach when hint missing")
    default_revenue: float = Field(default=0.5, ge=0, le=1, description="Default revenue when hint missing")
    default_risk_reduction: float = Field(default=0.5, ge=0, le=1, description="Default risk reduction when hint missing")
    default_engineering: float = Field(default=0.5, ge=0, le=1, description="Default engineering effort when hint missing")
    default_dependency: float = Field(default=0.5, ge=0, le=1, description="Default dependency complexity when hint missing")
    default_complexity: float = Field(default=0.5, ge=0, le=1, description="Default implementation complexity when hint missing")
    

    # Monitoring and observability configuration
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring and observability settings")
    
    # Keyword heuristics for automatic inference
    keyword_mappings: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            # High reach keywords
            "checkout": {"reach": 0.8, "revenue": 0.7},
            "search": {"reach": 0.9, "engineering": 0.6},
            "login": {"reach": 0.8, "engineering": 0.4},
            "signup": {"reach": 0.8, "revenue": 0.6},
            "homepage": {"reach": 0.9, "engineering": 0.5},
            "mobile": {"reach": 0.8, "engineering": 0.7},
            
            # High revenue keywords
            "pricing": {"revenue": 0.9, "engineering": 0.4},
            "payment": {"revenue": 0.9, "engineering": 0.6, "risk_reduction": 0.7},
            "subscription": {"revenue": 0.8, "engineering": 0.6},
            "upsell": {"revenue": 0.7, "engineering": 0.3},
            "conversion": {"revenue": 0.8, "reach": 0.6},
            "premium": {"revenue": 0.7, "engineering": 0.5},
            
            # High risk reduction keywords
            "security": {"risk_reduction": 0.9, "engineering": 0.7},
            "compliance": {"risk_reduction": 0.8, "engineering": 0.6},
            "backup": {"risk_reduction": 0.7, "engineering": 0.5},
            "monitoring": {"risk_reduction": 0.6, "engineering": 0.5},
            "audit": {"risk_reduction": 0.7, "engineering": 0.4},
            "encryption": {"risk_reduction": 0.8, "engineering": 0.6},
            
            # High engineering effort keywords
            "microservice": {"engineering": 0.8, "complexity": 0.7},
            "refactor": {"engineering": 0.7, "complexity": 0.6},
            "migration": {"engineering": 0.8, "dependency": 0.7, "complexity": 0.8},
            "infrastructure": {"engineering": 0.8, "dependency": 0.6},
            "architecture": {"engineering": 0.9, "complexity": 0.8},
            "api": {"engineering": 0.6, "dependency": 0.5},
            
            # High dependency keywords
            "integration": {"dependency": 0.8, "engineering": 0.6},
            "third-party": {"dependency": 0.7, "engineering": 0.5},
            "external": {"dependency": 0.7, "engineering": 0.5},
            "webhook": {"dependency": 0.6, "engineering": 0.4},
            "sync": {"dependency": 0.6, "engineering": 0.5},
            
            # High complexity keywords
            "rewrite": {"complexity": 0.9, "engineering": 0.8},
            "platform": {"complexity": 0.8, "engineering": 0.8},
            "framework": {"complexity": 0.7, "engineering": 0.7},
            "algorithm": {"complexity": 0.7, "engineering": 0.6},
            "optimization": {"complexity": 0.6, "engineering": 0.6},
            "machine-learning": {"complexity": 0.8, "engineering": 0.8},
            "ai": {"complexity": 0.8, "engineering": 0.8}
        },
        description="Keyword mappings for automatic factor inference"
    )
    
    @classmethod
    def default(cls) -> 'Config':
        """Get default configuration."""
        return cls()
    
    @classmethod
    def rice_config(cls) -> 'Config':
        """Get configuration optimized for RICE scoring."""
        return cls(prioritize_metric=ScoringPolicy.RICE)
    
    @classmethod
    def ice_config(cls) -> 'Config':
        """Get configuration optimized for ICE scoring."""
        return cls(prioritize_metric=ScoringPolicy.ICE)
    
    @classmethod
    def impact_effort_config(cls) -> 'Config':
        """Get configuration for Impact-Effort scoring."""
        return cls(prioritize_metric=ScoringPolicy.IMPACT_MINUS_EFFORT)
    
    @classmethod
    def llm_enabled_config(cls, policy: ScoringPolicy = ScoringPolicy.RICE) -> 'Config':
        """Get configuration with LLM enhancements enabled."""
        return cls(prioritize_metric=policy, llm_enabled=True)
    
    def get_scoring_params(self) -> Dict[str, Any]:
        """Get parameters for the current scoring policy."""
        return {
            "policy": self.prioritize_metric,
            "impact_weights": self.impact_weights.model_dump(),
            "effort_weights": self.effort_weights.model_dump()
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration parameters."""
        return {
            "enabled": self.llm_enabled,
            "model": self.llm_model,
            "api_key_env": self.llm_api_key_env,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "timeout": self.llm_timeout
        }
    
    def update_weights(self, impact_weights: Optional[Dict[str, float]] = None, effort_weights: Optional[Dict[str, float]] = None) -> 'Config':
        """Create new config with updated weights."""
        new_config = self.copy()
        
        if impact_weights:
            for key, value in impact_weights.items():
                if hasattr(new_config.impact_weights, key):
                    setattr(new_config.impact_weights, key, value)
        
        if effort_weights:
            for key, value in effort_weights.items():
                if hasattr(new_config.effort_weights, key):
                    setattr(new_config.effort_weights, key, value)
        
        return new_config
    

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration parameters."""
        return {
            "enabled": self.monitoring.enabled,
            "log_directory": self.monitoring.log_directory,
            "audit_trail_enabled": self.monitoring.audit_trail_enabled,
            "performance_metrics_enabled": self.monitoring.performance_metrics_enabled,
            "alert_thresholds": self.monitoring.alert_thresholds.model_dump(),
            "report_interval_minutes": self.monitoring.report_interval_minutes,
            "retention_days": self.monitoring.retention_days
        }
    
    @classmethod
    def monitoring_enabled_config(cls, policy: ScoringPolicy = ScoringPolicy.RICE) -> 'Config':
        """Get configuration with comprehensive monitoring enabled."""
        monitoring_config = MonitoringConfig(
            enabled=True,
            audit_trail_enabled=True,
            performance_metrics_enabled=True
        )
        return cls(prioritize_metric=policy, monitoring=monitoring_config)
    
    @classmethod
    def production_config(cls, policy: ScoringPolicy = ScoringPolicy.RICE) -> 'Config':
        """Get production-ready configuration with monitoring and LLM."""
        monitoring_config = MonitoringConfig(
            enabled=True,
            audit_trail_enabled=True,
            performance_metrics_enabled=True,
            log_directory="logs/production"
        )
        return cls(
            prioritize_metric=policy,
            llm_enabled=True,
            llm_model="gpt-4o-mini",
            monitoring=monitoring_config
        )