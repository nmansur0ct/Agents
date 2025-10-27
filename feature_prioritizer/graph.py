"""
Graph Module - LangGraph Orchestration
Implements the StateGraph with linear execution flow: Extract -> Score -> Prioritize.
"""

from typing import List, Optional
from langgraph.graph import StateGraph, END
from models import RawFeature, State
from nodes import extractor_node, scorer_node, prioritizer_node
from config import Config

# Import monitoring functionality
from monitoring import initialize_monitoring, get_system_monitor

class FeaturePrioritizationGraph:
    """LangGraph orchestration for feature prioritization pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the graph with optional configuration."""
        self.config = config or Config.default()
        

        # Initialize monitoring if enabled
        if self.config.monitoring.enabled:
            initialize_monitoring(self.config.model_dump())
        
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the state graph with nodes and edges."""
        # Create wrapper functions that capture the config
        def extractor_with_config(state: State) -> State:
            return extractor_node(state, self.config)
        
        def scorer_with_config(state: State) -> State:
            return scorer_node(state, self.config)
        
        def prioritizer_with_config(state: State) -> State:
            return prioritizer_node(state, self.config)
        
        # Create the state graph
        workflow = StateGraph(State)
        
        # Add nodes with config-aware wrappers
        workflow.add_node("extract", extractor_with_config)
        workflow.add_node("score", scorer_with_config)
        workflow.add_node("prioritize", prioritizer_with_config)
        
        # Define the flow: START -> extract -> score -> prioritize -> END
        workflow.set_entry_point("extract")
        workflow.add_edge("extract", "score")
        workflow.add_edge("score", "prioritize")
        workflow.add_edge("prioritize", END)
        
        return workflow.compile()
    
    def process_features(self, raw_features: List[RawFeature]) -> State:
        """
        Process a list of raw features through the complete pipeline.
        
        Args:
            raw_features: List of RawFeature objects to process
            
        Returns:
            Final state with prioritized results
        """
        # Initialize state
        initial_state: State = {
            'raw': raw_features,
            'errors': []
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state
    
    def get_prioritized_features(self, raw_features: List[RawFeature]) -> List[dict]:
        """
        Convenience method to get prioritized features as dictionaries.
        
        Args:
            raw_features: List of RawFeature objects to process
            
        Returns:
            List of prioritized features as dictionaries
        """
        final_state = self.process_features(raw_features)
        
        if 'prioritized' in final_state and final_state['prioritized']:
            return [feature.model_dump() for feature in final_state['prioritized'].ordered]
        
        return []
    
    def get_detailed_results(self, raw_features: List[RawFeature]) -> dict:
        """
        Get detailed results including all intermediate stages.
        
        Args:
            raw_features: List of RawFeature objects to process
            
        Returns:
            Dictionary with all stages and any errors
        """
        final_state = self.process_features(raw_features)
        
        results = {
            'input_count': len(raw_features),
            'errors': final_state.get('errors', []),
            'config': self.config.model_dump(),
            'stages': {}
        }
        
        # Add extraction results if available
        if 'extracted' in final_state and final_state['extracted']:
            results['stages']['extraction'] = {
                'feature_count': len(final_state['extracted'].features),
                'features': [f.model_dump() for f in final_state['extracted'].features]
            }
        
        # Add scoring results if available
        if 'scored' in final_state and final_state['scored']:
            results['stages']['scoring'] = {
                'feature_count': len(final_state['scored'].scored),
                'features': [f.model_dump() for f in final_state['scored'].scored]
            }
        
        # Add prioritization results if available
        if 'prioritized' in final_state and final_state['prioritized']:
            results['stages']['prioritization'] = {
                'feature_count': len(final_state['prioritized'].ordered),
                'features': [f.model_dump() for f in final_state['prioritized'].ordered]
            }
            
            # Add summary statistics
            scores = [f.score for f in final_state['prioritized'].ordered]
            if scores:
                results['summary'] = {
                    'highest_score': max(scores),
                    'lowest_score': min(scores),
                    'average_score': sum(scores) / len(scores),
                    'score_range': max(scores) - min(scores)
                }
        
        return results
    

    def get_monitoring_report(self):
        """Get comprehensive monitoring report if monitoring is enabled."""
        if self.config.monitoring.enabled:
            monitor = get_system_monitor()
            return monitor.generate_monitoring_report()
        return None
    
    def print_performance_summary(self):
        """Print performance summary to console if monitoring is enabled."""
        if self.config.monitoring.enabled:
            monitor = get_system_monitor()
            monitor.print_performance_summary()
        else:
            print("📊 Monitoring is disabled. Enable monitoring in config to see performance metrics.")
    
    def save_monitoring_report(self, filename: Optional[str] = None):
        """Save monitoring report to file if monitoring is enabled."""
        if self.config.monitoring.enabled:
            monitor = get_system_monitor()
            return monitor.save_monitoring_report(filename)
        return None

# Convenience function for simple usage
def prioritize_features(raw_features: List[RawFeature], config: Optional[Config] = None) -> List[dict]:
    """
    Simple function to prioritize features with default configuration.
    
    Args:
        raw_features: List of RawFeature objects
        config: Optional configuration (uses default if not provided)
        
    Returns:
        List of prioritized features as dictionaries
    """
    graph = FeaturePrioritizationGraph(config)
    return graph.get_prioritized_features(raw_features)