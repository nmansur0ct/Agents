#!/usr/bin/env python3
"""
Feature Prioritization Assistant - CLI Entry Point
Provides command-line interface for feature prioritization with JSON and file input support.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import click
import pandas as pd
from pydantic import ValidationError

from models import RawFeature
from graph import FeaturePrioritizationGraph
from config import Config, ScoringPolicy

@click.command()
@click.option(
    '--file', '-f',
    type=click.Path(exists=True, path_type=Path),
    help='Path to JSON or CSV file containing features'
)
@click.option(
    '--json', '-j',
    'json_input',  # This maps --json to the json_input parameter
    type=str,
    help='JSON string containing features array'
)
@click.option(
    '--metric', '-m',
    type=click.Choice(['RICE', 'ICE', 'ImpactMinusEffort'], case_sensitive=False),
    default='RICE',
    help='Prioritization metric to use (default: RICE)'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output file path (JSON format)'
)
@click.option(
    '--detailed', '-d',
    is_flag=True,
    help='Include detailed results with all stages'
)
@click.option(
    '--csv-output', '-c',
    type=click.Path(path_type=Path),
    help='Output results as CSV file'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output with processing details'
)
@click.option(
    '--llm',
    is_flag=True,
    help='Enable LLM enhancements for factor analysis and rationale generation'
)
@click.option(
    '--model',
    type=str,
    default='gpt-3.5-turbo',
    help='OpenAI model to use for LLM calls (default: gpt-3.5-turbo)'
)
@click.option(
    '--keyenv',
    type=str,
    default='OPENAI_API_KEY',
    help='Environment variable name for OpenAI API key (default: OPENAI_API_KEY)'
)
@click.option(
    '--auto-save',
    is_flag=True,
    help='Automatically save results with timestamp (JSON and CSV in results/ folder)'
)
@click.option(
    '--results-dir',
    type=str,
    default='results',
    help='Directory to save results (default: results)'
)
def main(file, json_input, metric, output, detailed, csv_output, verbose, llm, model, keyenv, auto_save, results_dir):
    """
    Feature Prioritization Assistant
    
    Prioritize product features based on impact and effort using configurable scoring policies.
    Optionally use LLM enhancements for intelligent factor analysis and rationale generation.
    
    Results are automatically saved with timestamps in the results/ folder for easy identification.
    
    Examples:
        # Basic prioritization with auto-save
        python run.py --file samples/features.json --metric RICE --auto-save
        
        # With LLM enhancements 
        python run.py --file features.json --metric RICE --llm --auto-save
        
        # Custom output locations (will be timestamped automatically)
        python run.py --file features.csv --output analysis.json --detailed
        python run.py --file features.json --csv-output summary.csv
        
        # Different models and configurations
        python run.py --file features.json --llm --model gpt-4o-mini --auto-save
    """
    
    try:
        # Validate input arguments
        if not file and not json_input:
            click.echo("Error: Must provide either --file or --json input", err=True)
            sys.exit(1)
        
        if file and json_input:
            click.echo("Error: Cannot use both --file and --json. Choose one.", err=True)
            sys.exit(1)
        
        # Load raw features
        raw_features = load_features(file, json_input, verbose)
        
        if not raw_features:
            click.echo("Error: No valid features found in input", err=True)
            sys.exit(1)
        
        if verbose:
            click.echo(f"Loaded {len(raw_features)} features")
        
        # Configure scoring policy with LLM settings
        config = get_config_for_metric(metric, llm, model, keyenv)
        
        if verbose:
            click.echo(f"Using {metric} scoring policy")
            if llm:
                click.echo(f"LLM enhancements enabled using {model}")
            else:
                click.echo("LLM enhancements disabled")
        
        # Process features
        graph = FeaturePrioritizationGraph(config)
        
        if detailed:
            results = graph.get_detailed_results(raw_features)
        else:
            prioritized_features = graph.get_prioritized_features(raw_features)
            results = {"prioritized_features": prioritized_features}
        
        # Handle any errors
        if 'errors' in results and results['errors']:
            click.echo("Warnings/Errors during processing:", err=True)
            for error in results['errors']:
                click.echo(f"  - {error}", err=True)
        
        # Output results
        saved_files = []
        
        # Handle auto-save option
        if auto_save:
            # Generate timestamped filenames in results directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metric_name = metric.lower()
            llm_suffix = "_llm" if llm else ""
            
            auto_json_path = Path(results_dir) / f"prioritization_{metric_name}{llm_suffix}_{timestamp}.json"
            auto_csv_path = Path(results_dir) / f"prioritization_{metric_name}{llm_suffix}_{timestamp}.csv"
            
            save_json_results(results, auto_json_path, verbose)
            save_csv_results(results, auto_csv_path, verbose)
            saved_files.extend([auto_json_path, auto_csv_path])
        
        # Handle explicit output options
        if output:
            # If relative path, put in results directory with timestamp
            if not output.is_absolute() and output.parent == Path('.'):
                stem = output.stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = Path(results_dir) / f"{stem}_{timestamp}.json"
            save_json_results(results, output, verbose)
            saved_files.append(output)
        
        if csv_output:
            # If relative path, put in results directory with timestamp
            if not csv_output.is_absolute() and csv_output.parent == Path('.'):
                stem = csv_output.stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_output = Path(results_dir) / f"{stem}_{timestamp}.csv"
            save_csv_results(results, csv_output, verbose)
            saved_files.append(csv_output)
        
        # Print to stdout if no output files specified and not auto-save
        if not output and not csv_output and not auto_save:
            print(json.dumps(results, indent=2))
        elif saved_files and verbose:
            click.echo(f"\nResults saved to:")
            for file_path in saved_files:
                click.echo(f"  📄 {file_path}")
        
        if verbose:
            if detailed and 'summary' in results:
                summary = results['summary']
                if isinstance(summary, dict):
                    click.echo(f"\nSummary:")
                    click.echo(f"  Highest score: {summary.get('highest_score', 0):.3f}")
                    click.echo(f"  Lowest score: {summary.get('lowest_score', 0):.3f}")
                    click.echo(f"  Average score: {summary.get('average_score', 0):.3f}")
                    click.echo(f"  Score range: {summary.get('score_range', 0):.3f}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

def load_features(file_path: Path, json_str: str, verbose: bool) -> List[RawFeature]:
    """Load features from file or JSON string."""
    raw_features = []
    
    try:
        if file_path:
            if file_path.suffix.lower() == '.csv':
                raw_features = load_features_from_csv(file_path, verbose)
            else:
                raw_features = load_features_from_json_file(file_path, verbose)
        
        elif json_str:
            raw_features = load_features_from_json_string(json_str, verbose)
    
    except Exception as e:
        raise click.ClickException(f"Failed to load features: {str(e)}")
    
    return raw_features

def load_features_from_json_file(file_path: Path, verbose: bool) -> List[RawFeature]:
    """Load features from JSON file."""
    if verbose:
        click.echo(f"Loading features from JSON file: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'features' in data:
        features_data = data['features']
    elif isinstance(data, list):
        features_data = data
    else:
        raise ValueError("JSON must contain 'features' array or be an array of features")
    
    return [RawFeature(**feature_data) for feature_data in features_data]

def load_features_from_json_string(json_str: str, verbose: bool) -> List[RawFeature]:
    """Load features from JSON string."""
    if verbose:
        click.echo("Loading features from JSON string")
    
    data = json.loads(json_str)
    
    if isinstance(data, list):
        features_data = data
    elif isinstance(data, dict) and 'features' in data:
        features_data = data['features']
    else:
        raise ValueError("JSON must be an array of features or contain 'features' array")
    
    return [RawFeature(**feature_data) for feature_data in features_data]

def load_features_from_csv(file_path: Path, verbose: bool) -> List[RawFeature]:
    """Load features from CSV file."""
    if verbose:
        click.echo(f"Loading features from CSV file: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Convert DataFrame to list of dictionaries
    features_data = df.to_dict('records')
    
    # Convert to RawFeature objects, handling missing columns
    raw_features = []
    for feature_data in features_data:
        # Clean up NaN values
        clean_data = {k: v for k, v in feature_data.items() if pd.notna(v)}
        
        # Ensure required fields
        if 'name' not in clean_data or 'description' not in clean_data:
            continue
        
        try:
            # Ensure clean_data has proper types
            feature_dict = {str(k): v for k, v in clean_data.items()}
            raw_features.append(RawFeature(**feature_dict))
        except ValidationError as e:
            if verbose:
                click.echo(f"Warning: Skipping invalid feature {clean_data.get('name', 'unknown')}: {e}")
    
    return raw_features

def get_config_for_metric(metric: str, llm_enabled: bool = False, llm_model: str = 'gpt-3.5-turbo', llm_api_key_env: str = 'OPENAI_API_KEY') -> Config:
    """Get configuration for the specified metric with optional LLM settings."""
    metric_upper = metric.upper()
    
    if metric_upper == 'RICE':
        config = Config.rice_config()
    elif metric_upper == 'ICE':
        config = Config.ice_config()
    elif metric_upper == 'IMPACTMINUSEFFORT':
        config = Config.impact_effort_config()
    else:
        config = Config.default()
    
    # Configure LLM settings if enabled
    if llm_enabled:
        config.llm_enabled = True
        config.llm_model = llm_model
        config.llm_api_key_env = llm_api_key_env
    
    return config

def generate_timestamped_path(base_name: str, extension: str, results_dir: str = "results") -> Path:
    """Generate timestamped file path in results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.{extension}"
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    return results_path / filename

def save_json_results(results: dict, output_path: Path, verbose: bool):
    """Save results to JSON file."""
    # If no specific path provided, generate timestamped path
    if output_path is None:
        output_path = generate_timestamped_path("prioritization_results", "json")
    elif not output_path.is_absolute() and output_path.parent == Path('.'):
        # If just a filename provided, put it in results directory with timestamp
        stem = output_path.stem
        suffix = output_path.suffix.lstrip('.')
        output_path = generate_timestamped_path(stem, suffix)
    
    if verbose:
        click.echo(f"Saving results to JSON file: {output_path}")
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def save_csv_results(results: dict, output_path: Path, verbose: bool):
    """Save prioritized features to CSV file."""
    # If no specific path provided, generate timestamped path
    if output_path is None:
        output_path = generate_timestamped_path("prioritization_results", "csv")
    elif not output_path.is_absolute() and output_path.parent == Path('.'):
        # If just a filename provided, put it in results directory with timestamp
        stem = output_path.stem
        suffix = output_path.suffix.lstrip('.')
        output_path = generate_timestamped_path(stem, suffix)
    
    if verbose:
        click.echo(f"Saving results to CSV file: {output_path}")
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract prioritized features
    if 'prioritized_features' in results:
        features = results['prioritized_features']
    elif 'stages' in results and 'prioritization' in results['stages']:
        features = results['stages']['prioritization']['features']
    else:
        raise ValueError("No prioritized features found in results")
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    
    # Save to CSV
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()