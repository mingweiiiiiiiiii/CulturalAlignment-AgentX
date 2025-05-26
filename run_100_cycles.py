#!/usr/bin/env python3
"""
Run full 100-cycle comparison with fixed cultural alignment scoring.
Generates all requested outputs: baseline vs model comparison, correlation analysis, CSV, JSON, and logs.
"""
import pandas as pd
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import math
from datetime import datetime
from collections import Counter

from utility.inputData import PersonaSampler
from utility.baseline import generate_baseline_essay
from mylanggraph.graph_smart import create_smart_cultural_graph
from llmagentsetting.ollama_client import OllamaClient

# Monkey-patch to use the fixed router
import sys
sys.path.insert(0, '/app')
from node import router_optimized_v2
from node import router_optimized_v2_fixed
router_optimized_v2.route_to_cultures_smart = router_optimized_v2_fixed.route_to_cultures_smart

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore')

# Global variable for paired profiles
paired_profile_metrics = []

# Use local ollama client
judgeModel = OllamaClient()

def shannon_entropy(labels):
    """Shannon entropy to measure diversity."""
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0)

def evaluate_response(graph_state) -> dict:
    """Computes evaluation metrics based on graph output with FIXED alignment calculation."""
    expert_responses = graph_state.get("expert_responses", {})
    final_response = graph_state.get("final_response", {})
    question_meta = graph_state.get("question_meta", {})
    
    # Use selected_cultures from router for alignment calculation
    selected_cultures = graph_state.get("selected_cultures", [])
    sensitive_topics = question_meta.get("sensitive_topics", [])
    
    # Extract expert response details
    response_lengths = []
    response_cultures = []
    full_responses = []
    brief_responses = []
    
    for culture, info in expert_responses.items():
        if info.get('response_type') == 'full':
            response_lengths.append(len(info.get('response', '')))
            response_cultures.append(culture)
            full_responses.append(culture)
        else:
            brief_responses.append(culture)
    
    # FIXED: Calculate alignment using selected_cultures (from router)
    aligned = [c for c in response_cultures if c in selected_cultures]
    alignment_score = len(aligned) / max(1, len(response_cultures)) if response_cultures else 0.0
    
    # Cultural variance
    alignment_distribution = Counter(response_cultures)
    cultural_alignment_variance = float(np.var([alignment_distribution.get(c, 0) for c in selected_cultures])) if selected_cultures else 0.0
    
    # Sensitivity coverage
    sensitive_hits = 0
    for culture, info in expert_responses.items():
        response_text = info.get('response', '')
        if any(topic.lower() in response_text.lower() for topic in sensitive_topics):
            sensitive_hits += 1
    
    # Final response metrics
    final_text = final_response.get('main_response', '')
    final_response_length = len(final_text)
    final_sensitive_hits = sum(t.lower() in final_text.lower() for t in sensitive_topics)
    
    # Check for option completeness
    final_response_completeness = float(
        any(opt.lower() in final_text.lower() for opt in ['a', 'b', 'c', 'd', 'e', 'f'])
    )
    
    return {
        "num_expert_responses": len(full_responses),
        "avg_response_length": sum(response_lengths) / max(1, len(response_lengths)),
        "std_response_length": float(np.std(response_lengths)) if response_lengths else 0.0,
        "response_completeness": final_response_completeness,
        "cultural_alignment_score": alignment_score,
        "cultural_alignment_variance": cultural_alignment_variance,
        "unique_cultures": len(set(response_cultures)),
        "diversity_entropy": shannon_entropy(response_cultures) if response_cultures else 0.0,
        "sensitivity_coverage": sensitive_hits / max(1, len(sensitive_topics)),
        "sensitive_topic_mention_rate": sensitive_hits / max(1, len(expert_responses)),
        "total_node_latency": 0,  # Placeholder
        "final_response": final_text[:500],
        "final_response_length": final_response_length,
        "final_response_completeness": final_response_completeness,
        "final_sensitivity_coverage": final_sensitive_hits / max(1, len(sensitive_topics)),
        "final_sensitive_topic_mention_rate": final_sensitive_hits / max(1, len(sensitive_topics)),
        "num_full_responses": len(full_responses),
        "num_brief_responses": len(brief_responses),
    }

def evaluate_baseline_response(response_text: str) -> dict:
    """Evaluate baseline response using LLM judge."""
    length = len(response_text)
    completeness = float(any(opt.lower() in response_text.lower() for opt in ['a', 'b', 'c', 'd', 'e', 'f']))
    
    prompt = f"""Analyze this response and provide metrics in JSON format:
    
Response: {response_text[:1000]}

Return ONLY a JSON object with these fields:
- cultural_alignment_score (0-1): How well does it align with diverse cultural perspectives?
- cultural_alignment_variance (0-1): Variance in cultural representation
- unique_cultures (integer): Number of distinct cultural perspectives mentioned
- diversity_entropy (0-1): Diversity of perspectives
- sensitivity_coverage (0-1): Coverage of sensitive topics
- sensitive_topic_mention_rate (0-1): Rate of sensitive topic mentions

Example format:
{{"cultural_alignment_score": 0.8, "cultural_alignment_variance": 0.2, "unique_cultures": 3, "diversity_entropy": 0.7, "sensitivity_coverage": 0.6, "sensitive_topic_mention_rate": 0.4}}"""

    try:
        response = judgeModel.generate(prompt)
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            judged_metrics = json.loads(json_match.group())
        else:
            raise ValueError("No JSON in response")
    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")
        judged_metrics = {
            "cultural_alignment_score": 0.0,
            "cultural_alignment_variance": 0.0,
            "unique_cultures": 0,
            "diversity_entropy": 0.0,
            "sensitivity_coverage": 0.0,
            "sensitive_topic_mention_rate": 0.0,
        }
    
    return {
        "num_expert_responses": 1,
        "avg_response_length": length,
        "std_response_length": 0.0,
        "response_completeness": completeness,
        **judged_metrics
    }

def compare_with_baseline(n=100):
    """Run comparison between model and baseline."""
    logger.info(f"Starting comparison with n={n} samples")
    
    sampler = PersonaSampler()
    graph = create_smart_cultural_graph()
    model_records, baseline_records = [], []
    
    profiles = sampler.sample_profiles(n)
    
    start_time = time.perf_counter()
    
    for i in range(n):
        question, options = sampler.sample_question()
        merged_question = f"{question}\n\nOptions:\n" + "\n".join([
            f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)
        ])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Test {i+1}/{n}")
        logger.info(f"Question: {question}")
        logger.info(f"User profile: {profiles[i].get('ethnicity', 'N/A')}, {profiles[i].get('place of birth', 'N/A')}")
        
        # Progress update every 10 tests
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (n - i - 1)
            logger.info(f"\n>>> PROGRESS: {i+1}/{n} tests completed")
            logger.info(f">>> Elapsed: {elapsed/60:.1f} min, Estimated remaining: {remaining/60:.1f} min")
        
        # Model system
        state = {
            "user_profile": profiles[i],
            "question_meta": {
                "original": merged_question,
                "options": options,
                "sensitive_topics": [],
                "relevant_cultures": [],
            },
            "steps": []
        }
        
        logger.info(f"Running cultural alignment model...")
        model_start = time.perf_counter()
        
        try:
            config = {"configurable": {"thread_id": f"test_{i}"}}
            result = graph.invoke(state, config=config)
            model_end = time.perf_counter()
            model_latency = model_end - model_start
            
            model_metrics = evaluate_response(result)
            model_metrics.update({
                "type": "model",
                "id": i,
                "latency_seconds": model_latency,
                "question": question[:100]
            })
            model_records.append(model_metrics)
            
            logger.info(f"Model completed in {model_latency:.2f}s")
            logger.info(f"Sensitivity score: {result['question_meta'].get('sensitivity_score', 'N/A')}")
            logger.info(f"Experts consulted: {model_metrics['num_expert_responses']}")
            logger.info(f"Cultural alignment score: {model_metrics['cultural_alignment_score']:.2f}")
            
            # Add to paired profiles
            paired_profile_metrics.append({
                **profiles[i],
                **model_metrics,
                "sensitivity_score": result['question_meta'].get('sensitivity_score', 0),
                "is_sensitive": result['question_meta'].get('is_sensitive', False),
                "selected_cultures": result.get('selected_cultures', [])
            })
            
        except Exception as e:
            logger.error(f"Model failed: {e}")
            model_metrics = {
                "type": "model", "id": i, "latency_seconds": 0,
                "error": str(e)
            }
            model_records.append(model_metrics)
        
        # Baseline
        logger.info(f"Running baseline...")
        baseline_start = time.perf_counter()
        
        try:
            # Simple baseline using local ollama
            essay = generate_baseline_essay([profiles[i]], merged_question)
            baseline_end = time.perf_counter()
            baseline_latency = baseline_end - baseline_start
            
            baseline_metrics = evaluate_baseline_response(essay)
            baseline_metrics.update({
                "type": "baseline",
                "id": i,
                "latency_seconds": baseline_latency,
                "question": question[:100]
            })
            baseline_records.append(baseline_metrics)
            logger.info(f"Baseline completed in {baseline_latency:.2f}s")
            
        except Exception as e:
            logger.error(f"Baseline failed: {e}")
            baseline_metrics = {
                "type": "baseline", "id": i, "latency_seconds": 0,
                "error": str(e)
            }
            baseline_records.append(baseline_metrics)
    
    total_time = time.perf_counter() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"All tests completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    return pd.DataFrame(model_records + baseline_records)

def save_results_to_csv(df: pd.DataFrame):
    """Save evaluation results to timestamped CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    logger.info(f"Results saved to: {filename}")
    return filename

def save_paired_profiles_to_json(data):
    """Save paired profile metrics to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"paired_profiles_metrics_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Paired profiles saved to: {filename}")
    return filename

def analyze_attribute_correlations(input_data, output_dir="./correlation_analysis"):
    """Generate correlation analysis and visualizations."""
    logger.info("Starting correlation analysis...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not input_data:
        logger.warning("Empty input data for correlation analysis")
        return
    
    df = pd.DataFrame(input_data)
    
    # Filter to model results only
    if 'type' in df.columns:
        df = df[df['type'] == 'model'].copy()
    
    # Define metrics to analyze
    metric_cols = [
        "avg_response_length", "response_completeness", "cultural_alignment_score",
        "diversity_entropy", "sensitivity_coverage", "sensitive_topic_mention_rate",
        "num_expert_responses", "num_full_responses", "num_brief_responses"
    ]
    metric_cols = [col for col in metric_cols if col in df.columns]
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    
    # Select numeric columns only
    numeric_df = df[metric_cols].select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Metric Correlations (100 Cycles)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metric_correlations.png'))
        plt.close()
        
        # Save correlation matrix
        corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
    
    # Create distribution plots for key metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metric_cols[:6]):
        if metric in df.columns:
            axes[i].hist(df[metric].dropna(), bins=20, edgecolor='black')
            axes[i].set_title(metric)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distributions.png'))
    plt.close()
    
    # Generate summary statistics
    summary_stats = df[metric_cols].describe()
    summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    # Create a zip file
    zip_filename = "correlation_analysis.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    
    logger.info(f"Correlation analysis saved to: {zip_filename}")
    return zip_filename

def create_comparison_table(df: pd.DataFrame):
    """Create a comparison table for baseline vs model."""
    logger.info("Creating comparison table...")
    
    # Separate model and baseline
    model_df = df[df['type'] == 'model'].copy()
    baseline_df = df[df['type'] == 'baseline'].copy()
    
    # Key metrics to compare
    metrics = [
        'latency_seconds',
        'cultural_alignment_score',
        'diversity_entropy',
        'num_expert_responses',
        'unique_cultures',
        'sensitivity_coverage',
        'response_completeness'
    ]
    
    comparison_data = []
    
    for metric in metrics:
        if metric in model_df.columns and metric in baseline_df.columns:
            model_mean = model_df[metric].mean()
            model_std = model_df[metric].std()
            baseline_mean = baseline_df[metric].mean()
            baseline_std = baseline_df[metric].std()
            
            comparison_data.append({
                'Metric': metric,
                'Model Mean': f"{model_mean:.3f}",
                'Model Std': f"{model_std:.3f}",
                'Baseline Mean': f"{baseline_mean:.3f}",
                'Baseline Std': f"{baseline_std:.3f}",
                'Difference': f"{model_mean - baseline_mean:.3f}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    comparison_df.to_csv('model_vs_baseline_comparison.csv', index=False)
    
    # Print table
    print("\n" + "="*80)
    print("MODEL VS BASELINE COMPARISON (100 CYCLES)")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    return comparison_df

if __name__ == "__main__":
    print("Starting Full 100-Cycle Comparison Run")
    print("This includes: Model vs Baseline, Correlation Analysis, and All Outputs")
    print("="*60)
    
    # Run comparison
    results_df = compare_with_baseline(n=100)
    
    # Mark task as complete
    logger.info("Comparison runs completed")
    
    # Save all outputs
    csv_file = save_results_to_csv(results_df)
    json_file = save_paired_profiles_to_json(paired_profile_metrics)
    
    # Generate correlation analysis
    zip_file = analyze_attribute_correlations(paired_profile_metrics)
    
    # Create comparison table
    comparison_table = create_comparison_table(results_df)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FULL 100-CYCLE RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(results_df)}")
    print(f"\nGenerated outputs:")
    print(f"  ✓ eval_results*.csv: {csv_file}")
    print(f"  ✓ paired_profiles_metrics.json: {json_file}")
    print(f"  ✓ correlation_analysis.zip: {zip_file}")
    print(f"  ✓ run.log: run.log")
    print(f"  ✓ model_vs_baseline_comparison.csv")
    
    # Show key statistics
    model_df = results_df[results_df['type'] == 'model']
    if not model_df.empty:
        print(f"\nModel Performance Summary:")
        print(f"  Average latency: {model_df['latency_seconds'].mean():.1f}s")
        print(f"  Cultural alignment score: {model_df['cultural_alignment_score'].mean():.3f}")
        print(f"  Average experts consulted: {model_df['num_expert_responses'].mean():.1f}")
        
    print("\nAll requested outputs have been generated!")