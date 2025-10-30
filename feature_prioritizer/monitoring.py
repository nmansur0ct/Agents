"""

Monitoring and Observability Module
Implements comprehensive tracking, metrics, and audit trail for the agentic feature prioritization system.
Based on the monitoring framework documented in FEATURE_PRIORITY_ARCHITECTURE.md
"""

import logging
import time
import hashlib
import json
import gc
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
from functools import wraps
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

from models import (
    State, RawFeature, FeatureSpec, ScoredFeature,
    AgentExecutionRecord, SystemHealthMetrics, AgentPerformanceMetrics,
    AuditLogEntry, MonitoringReport, AlertThresholds, MonitoringConfig
)

class AgentMetrics:
    """
    Comprehensive metrics collection and analysis for agent performance.
    Implements the monitoring framework documented in FEATURE_PRIORITY_ARCHITECTURE.md
    """
    
    def __init__(self):
        self.processing_times = defaultdict(list)
        self.success_rates = defaultdict(lambda: {"success": 0, "failure": 0})
        self.llm_usage = defaultdict(lambda: {"calls": 0, "tokens": 0})
        self.executions: List[AgentExecutionRecord] = []
        self.start_time = datetime.now()
    
    def record_agent_execution(self, agent_name: str, duration: float, success: bool, 
                             llm_calls: int = 0, llm_tokens: int = 0, 
                             error_message: Optional[str] = None,
                             input_size: int = 0, output_size: int = 0,
                             confidence_score: Optional[float] = None):
        """Record comprehensive agent execution metrics."""
        self.processing_times[agent_name].append(duration)
        status = "success" if success else "failure"
        self.success_rates[agent_name][status] += 1
        
        if llm_calls > 0:
            self.llm_usage[agent_name]["calls"] += llm_calls
            self.llm_usage[agent_name]["tokens"] += llm_tokens
        
        # Store detailed execution record
        execution = AgentExecutionRecord(
            agent_name=agent_name,
            action="process",
            timestamp=datetime.now(),
            duration=duration,
            success=success,
            input_size=input_size,
            output_size=output_size,
            llm_calls=llm_calls,
            llm_tokens=llm_tokens,
            error_message=error_message,
            confidence_score=confidence_score
        )
        self.executions.append(execution)
    
    def get_performance_summary(self) -> Dict[str, AgentPerformanceMetrics]:
        """Generate comprehensive performance summary for all agents."""
        summary = {}
        
        for agent in self.processing_times.keys():
            times = self.processing_times[agent]
            rates = self.success_rates[agent]
            llm_stats = self.llm_usage[agent]
            
            total_executions = rates["success"] + rates["failure"]
            
            if total_executions > 0:
                summary[agent] = AgentPerformanceMetrics(
                    agent_name=agent,
                    avg_duration=float(np.mean(times)) if times else 0.0,
                    min_duration=float(np.min(times)) if times else 0.0,
                    max_duration=float(np.max(times)) if times else 0.0,
                    total_executions=total_executions,
                    success_rate=rates["success"] / total_executions,
                    error_rate=rates["failure"] / total_executions,
                    total_llm_calls=llm_stats["calls"],
                    total_llm_tokens=llm_stats["tokens"],
                    avg_tokens_per_call=llm_stats["tokens"] / llm_stats["calls"] if llm_stats["calls"] > 0 else 0,
                    mean_time_between_failures=self._calculate_mtbf(agent)
                )
        
        return summary
    
    def _calculate_mtbf(self, agent_name: str) -> float:
        """Calculate Mean Time Between Failures for an agent."""
        agent_executions = [e for e in self.executions if e.agent_name == agent_name]
        if not agent_executions:
            return float('inf')
        
        failure_times = [e.timestamp for e in agent_executions if not e.success]
        if len(failure_times) < 2:
            return float('inf')  # No failures or only one failure
        
        intervals = []
        for i in range(1, len(failure_times)):
            interval = (failure_times[i] - failure_times[i-1]).total_seconds()
            intervals.append(interval)
        
        return float(np.mean(intervals)) if intervals else float('inf')
    
    def get_system_health(self) -> SystemHealthMetrics:
        """Get overall system health metrics."""
        total_executions = sum(
            rates["success"] + rates["failure"] 
            for rates in self.success_rates.values()
        )
        
        total_successes = sum(
            rates["success"] 
            for rates in self.success_rates.values()
        )
        
        total_llm_calls = sum(
            usage["calls"] 
            for usage in self.llm_usage.values()
        )
        
        total_llm_tokens = sum(
            usage["tokens"] 
            for usage in self.llm_usage.values()
        )
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return SystemHealthMetrics(
            uptime_seconds=uptime,
            total_executions=total_executions,
            overall_success_rate=total_successes / total_executions if total_executions > 0 else 1.0,
            executions_per_minute=(total_executions / uptime) * 60 if uptime > 0 else 0.0,
            total_llm_calls=total_llm_calls,
            total_llm_tokens=total_llm_tokens,
            avg_tokens_per_execution=total_llm_tokens / total_executions if total_executions > 0 else 0.0
        )

class AuditTrail:
    """
    Comprehensive audit trail implementation for agent decisions and actions.
    Provides full transparency and traceability for all system operations.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        self.audit_entries: List[AuditLogEntry] = []
        self.log_file = log_file or Path("logs/audit_trail.jsonl")
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup structured logging
        self.audit_logger = self._setup_audit_logger()
    
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup dedicated audit logger with structured formatting."""
        logger = logging.getLogger("agent_audit")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler for audit trail
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "extra": %(extra)s}'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.propagate = False  # Prevent propagation to root logger
        
        return logger
    
    def create_audit_entry(self, agent_name: str, action: str, input_data: Any, 
                          output_data: Any, config: Dict[str, Any], 
                          execution_time: float, success: bool = True, 
                          error_details: Optional[str] = None) -> AuditLogEntry:
        """Create comprehensive audit trail entry for agent decisions."""
        
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            agent_name=agent_name,
            action=action,
            input_hash=self._hash_data(input_data),
            output_hash=self._hash_data(output_data),
            config_snapshot=self._serialize_config(config),
            decision_factors=self._extract_decision_factors(output_data),
            confidence_score=self._calculate_confidence(input_data, output_data),
            execution_time=execution_time,
            success=success,
            error_details=error_details
        )
        
        self.audit_entries.append(entry)
        
        # Log to structured audit log
        self._log_audit_entry(entry)
        
        return entry
    
    def _hash_data(self, data: Any) -> str:
        """Create hash of data for integrity checking."""
        try:
            if hasattr(data, 'model_dump'):
                # Pydantic model
                serialized = json.dumps(data.model_dump(), sort_keys=True, default=str)
            elif hasattr(data, '__dict__'):
                # Object with attributes
                serialized = json.dumps(data.__dict__, sort_keys=True, default=str)
            elif isinstance(data, dict):
                # Dictionary
                serialized = json.dumps(data, sort_keys=True, default=str)
            else:
                # Fallback to string representation
                serialized = str(data)
            
            return hashlib.md5(serialized.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(data).encode()).hexdigest()
    
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
    def _serialize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize configuration for audit trail."""
        return {
            key: value for key, value in config.items() 
            if not callable(value) and not key.startswith('_')
        }
    
    def _extract_decision_factors(self, output_data: Any) -> Dict[str, Any]:
        """Extract key decision factors from agent output."""
        factors = {}
        
        if hasattr(output_data, 'features'):
            # ExtractorOutput
            features = output_data.features
            if features:
                factors["num_features"] = len(features)
                factors["avg_reach"] = float(np.mean([f.reach for f in features]))
                factors["avg_revenue"] = float(np.mean([f.revenue for f in features]))
                factors["avg_engineering"] = float(np.mean([f.engineering for f in features]))
                factors["features_with_notes"] = sum(1 for f in features if f.notes)
        
        elif hasattr(output_data, 'scored'):
            # ScorerOutput
            scored = output_data.scored
            if scored:
                factors["num_scored"] = len(scored)
                factors["avg_impact"] = float(np.mean([f.impact for f in scored]))
                factors["avg_effort"] = float(np.mean([f.effort for f in scored]))
                scores = [f.score for f in scored]
                factors["score_range"] = [float(min(scores)), float(max(scores))]
        
        elif hasattr(output_data, 'ordered'):
            # PrioritizedOutput
            ordered = output_data.ordered
            if ordered:
                factors["num_prioritized"] = len(ordered)
                factors["top_score"] = float(ordered[0].score) if ordered else 0
                factors["score_distribution"] = self._calculate_score_distribution(ordered)
        
        return factors
    
    def _calculate_score_distribution(self, features: List[ScoredFeature]) -> Dict[str, float]:
        """Calculate distribution statistics for scores."""
        if not features:
            return {}
        
        scores = [f.score for f in features]
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "p25": float(np.percentile(scores, 25)),
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75))
        }
    
    def _calculate_confidence(self, input_data: Any, output_data: Any) -> float:
        """Calculate confidence score for agent decision."""
        # Base confidence on data quality and consistency
        confidence = 0.5  # Default moderate confidence
        
        # Increase confidence based on input data quality
        if hasattr(input_data, '__len__') and len(input_data) > 0:
            confidence += 0.2
        
        # Increase confidence based on output data completeness
        if hasattr(output_data, 'features') and output_data.features:
            # Check if features have notes (indicates LLM enhancement)
            features_with_notes = sum(1 for f in output_data.features if f.notes)
            if features_with_notes > 0:
                confidence += 0.2
        
        # Check for LLM analysis indicators
        if hasattr(output_data, 'scored') and output_data.scored:
            # Check for rationale (indicates thoughtful analysis)
            rationale_quality = sum(1 for f in output_data.scored if len(f.rationale) > 50)
            if rationale_quality > 0:
                confidence += 0.1
        
        # Ensure confidence is in [0, 1] range
        return min(1.0, max(0.0, confidence))
    
    def _log_audit_entry(self, entry: AuditLogEntry):
        """Log audit entry to structured log file."""
        extra_data = {
            "agent": entry.agent_name,
            "action": entry.action,
            "success": entry.success,
            "execution_time": entry.execution_time,
            "confidence": entry.confidence_score,
            "decision_factors": entry.decision_factors,
            "input_hash": entry.input_hash,
            "output_hash": entry.output_hash
        }
        
        if entry.success:
            self.audit_logger.info(
                f"Agent {entry.agent_name} completed {entry.action}",
                extra={"extra": json.dumps(extra_data, default=str)}
            )
        else:
            extra_data["error"] = entry.error_details
            self.audit_logger.error(
                f"Agent {entry.agent_name} failed {entry.action}: {entry.error_details}",
                extra={"extra": json.dumps(extra_data, default=str)}
            )
    
    def get_audit_summary(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of audit trail entries."""
        entries = self.audit_entries
        if agent_name:
            entries = [e for e in entries if e.agent_name == agent_name]
        
        if not entries:
            return {"total_entries": 0}
        
        return {
            "total_entries": len(entries),
            "success_rate": sum(1 for e in entries if e.success) / len(entries),
            "avg_execution_time": float(np.mean([e.execution_time for e in entries])),
            "avg_confidence": float(np.mean([e.confidence_score for e in entries])),
            "time_range": {
                "start": min(e.timestamp for e in entries).isoformat(),
                "end": max(e.timestamp for e in entries).isoformat()
            },
            "agents_involved": list(set(e.agent_name for e in entries))
        }

def audited_agent_execution(agent_name: str, action: str, metrics: AgentMetrics, 
                           audit_trail: AuditTrail, config: Dict[str, Any]):
    """
    Decorator for comprehensive agent execution monitoring and auditing.
    Implements the monitoring framework documented in FEATURE_PRIORITY_ARCHITECTURE.md
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            input_data = _capture_input_context(args, kwargs)
            llm_calls = 0
            llm_tokens = 0
            
            try:
                # Execute the agent function
                result = func(*args, **kwargs)
                
                # Calculate execution metrics
                execution_time = time.time() - start_time
                
                # Extract LLM usage if available in result
                if hasattr(result, 'get') and isinstance(result, dict):
                    llm_calls = result.get('llm_calls', 0)
                    llm_tokens = result.get('llm_tokens', 0)
                
                # Calculate data sizes
                input_size = _calculate_data_size(input_data)
                output_size = _calculate_data_size(result)
                
                # Record metrics
                metrics.record_agent_execution(
                    agent_name=agent_name,
                    duration=execution_time,
                    success=True,
                    llm_calls=llm_calls,
                    llm_tokens=llm_tokens,
                    input_size=input_size,
                    output_size=output_size
                )
                
                # Create audit entry
                audit_trail.create_audit_entry(
                    agent_name=agent_name,
                    action=action,
                    input_data=input_data,
                    output_data=result,
                    config=config,
                    execution_time=execution_time,
                    success=True
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_message = str(e)
                
                # Record failure metrics
                metrics.record_agent_execution(
                    agent_name=agent_name,
                    duration=execution_time,
                    success=False,
                    error_message=error_message,
                    input_size=_calculate_data_size(input_data)
                )
                
                # Create failure audit entry
                audit_trail.create_audit_entry(
                    agent_name=agent_name,
                    action=f"{action}_failed",
                    input_data=input_data,
                    output_data=error_message,
                    config=config,
                    execution_time=execution_time,
                    success=False,
                    error_details=error_message
                )
                
                raise  # Re-raise the exception
        
        return wrapper
    return decorator

def _capture_input_context(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Capture input context for audit trail."""
    context = {}
    
    # Extract state from args (typically first argument)
    if args and hasattr(args[0], 'get'):
        state = args[0]
        context["input_state"] = {
            "has_raw": "raw" in state,
            "has_extracted": "extracted" in state,
            "has_scored": "scored" in state,
            "has_prioritized": "prioritized" in state,
            "num_errors": len(state.get("errors", []))
        }
        
        # Extract feature count from raw data
        if "raw" in state and state["raw"]:
            context["num_input_features"] = len(state["raw"])
    
    # Capture kwargs (but avoid sensitive data)
    context["kwargs"] = {k: str(v) for k, v in kwargs.items() if not k.startswith('_')}
    
    return context

def _calculate_data_size(data: Any) -> int:
    """Calculate approximate size of data for monitoring."""
    try:
        if hasattr(data, 'model_dump'):
            # Pydantic model
            return len(json.dumps(data.model_dump(), default=str))
        elif isinstance(data, dict):
            return len(json.dumps(data, default=str))
        elif isinstance(data, (list, tuple)):
            return len(str(data))
        else:
            return len(str(data))
    except Exception:
        return 0

class SystemMonitor:
    """
    High-level system monitoring that coordinates metrics and audit trail.
    Provides unified interface for all observability features.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = AgentMetrics()
        self.audit_trail = AuditTrail(self.log_dir / "audit_trail.jsonl")
        self.config_snapshot = {}
        self.enabled = True
    
    def initialize_monitoring(self, config: Dict[str, Any]):
        """Initialize monitoring with system configuration."""
        self.config_snapshot = config.copy()
        self.enabled = config.get('monitoring', {}).get('enabled', True)
        
        if self.enabled:
            # Log system startup
            self.audit_trail.create_audit_entry(
                agent_name="system",
                action="startup",
                input_data=config,
                output_data="System monitoring initialized",
                config=config,
                execution_time=0.0,
                success=True
            )
            
            print("🔍 Monitoring system initialized")
            print(f"📁 Log directory: {self.log_dir}")
            print(f"📊 Metrics collection: {'Enabled' if self.enabled else 'Disabled'}")
    
    def get_monitoring_decorator(self, agent_name: str, action: str):
        """Get monitoring decorator for specific agent."""
        if not self.enabled:
            # Return pass-through decorator if monitoring disabled
            def disabled_decorator(func):
                return func
            return disabled_decorator
        
        return audited_agent_execution(
            agent_name=agent_name,
            action=action,
            metrics=self.metrics,
            audit_trail=self.audit_trail,
            config=self.config_snapshot
        )
    
    def generate_monitoring_report(self) -> MonitoringReport:
        """Generate comprehensive monitoring report."""
        return MonitoringReport(
            report_timestamp=datetime.now(),
            system_health=self.metrics.get_system_health(),
            agent_performance=self.metrics.get_performance_summary(),
            audit_summary=self.audit_trail.get_audit_summary(),
            recommendations=self._generate_recommendations()
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        system_health = self.metrics.get_system_health()
        agent_performance = self.metrics.get_performance_summary()
        
        # Check overall success rate
        if system_health.overall_success_rate < 0.95:
            recommendations.append(
                f"⚠️ Overall success rate is {system_health.overall_success_rate:.1%}. "
                "Consider investigating frequent failures."
            )
        
        # Check individual agent performance
        for agent_name, perf in agent_performance.items():
            if perf.error_rate > 0.05:
                recommendations.append(
                    f"⚠️ {agent_name} has high error rate ({perf.error_rate:.1%}). "
                    "Review error logs and improve error handling."
                )
            
            if perf.avg_duration > 10.0:
                recommendations.append(
                    f"🐌 {agent_name} has slow average duration ({perf.avg_duration:.1f}s). "
                    "Consider optimizing processing logic or LLM timeouts."
                )
        
        # Check LLM usage efficiency
        if system_health.avg_tokens_per_execution > 1000:
            recommendations.append(
                f"💰 High LLM token usage ({system_health.avg_tokens_per_execution:.0f} tokens/execution). "
                "Consider optimizing prompts or reducing LLM calls."
            )
        
        if not recommendations:
            recommendations.append("✅ All metrics are within healthy ranges. System performing well!")
        
        return recommendations
    
    def save_monitoring_report(self, filename: Optional[str] = None) -> Path:
        """Save monitoring report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_report_{timestamp}.json"
        
        # Handle absolute paths vs relative paths
        if Path(filename).is_absolute():
            report_path = Path(filename)
        else:
            report_path = self.log_dir / filename
        
        # Ensure parent directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_monitoring_report()
        
        with open(report_path, 'w') as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        
        return report_path
    
    def print_performance_summary(self):
        """Print a concise performance summary to console."""
        if not self.enabled:
            print("📊 Monitoring disabled")
            return
        
        system_health = self.metrics.get_system_health()
        agent_performance = self.metrics.get_performance_summary()
        
        print("\n" + "="*60)
        print("📊 AGENT PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"🔄 Total Executions: {system_health.total_executions}")
        print(f"✅ Success Rate: {system_health.overall_success_rate:.1%}")
        print(f"⏱️  Uptime: {system_health.uptime_seconds:.1f}s")
        print(f"🚀 Throughput: {system_health.executions_per_minute:.1f} exec/min")
        
        if system_health.total_llm_calls > 0:
            print(f"🤖 LLM Calls: {system_health.total_llm_calls}")
            print(f"💬 Total Tokens: {system_health.total_llm_tokens:,}")
            print(f"📏 Avg Tokens/Exec: {system_health.avg_tokens_per_execution:.0f}")
        
        print("\n" + "-"*40)
        print("AGENT DETAILS:")
        print("-"*40)
        
        for agent_name, perf in agent_performance.items():
            print(f"🤖 {agent_name}:")
            print(f"   ⏱️  Avg Duration: {perf.avg_duration:.2f}s")
            print(f"   ✅ Success Rate: {perf.success_rate:.1%}")
            print(f"   🔢 Executions: {perf.total_executions}")
            if perf.total_llm_calls > 0:
                print(f"   🤖 LLM Calls: {perf.total_llm_calls}")
        
        print("="*60 + "\n")

# Global monitoring instance for easy access
_global_monitor: Optional[SystemMonitor] = None

def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor

def initialize_monitoring(config: Dict[str, Any], log_dir: Optional[Path] = None):
    """Initialize global monitoring system."""
    global _global_monitor
    _global_monitor = SystemMonitor(log_dir)
    _global_monitor.initialize_monitoring(config)
    return _global_monitor

def reset_monitoring():
    """Reset monitoring system (useful for testing)."""
    global _global_monitor
    _global_monitor = None