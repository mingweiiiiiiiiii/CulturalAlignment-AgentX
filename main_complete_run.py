"""
Complete run of main.py with all outputs for 10 inputs.
Generates:
- correlation_analysis/ directory (to be zipped)
- eval_results_*.csv
- run.log
- paired_profiles_metrics_*.json
"""
import json
import math
import os
import time
import sys
from collections import Counter
from datetime import datetime
import logging
import zipfile
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from llmagentsetting.ollama_client import OllamaClient
from mylanggraph.graph_smart import create_smart_cultural_graph
from utility.baseline_local import generate_baseline_essay
from utility.inputData import PersonaSampler

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
    """Computes evaluation metrics based on graph output."""
    # Extract from new graph structure
    expert_responses = graph_state.get("expert_responses", {})
    final_response = graph_state.get("final_response", {})
    question_meta = graph_state.get("question_meta", {})
    
    # Get relevant cultures and topics
    relevant_cultures = question_meta.get("relevant_cultures", [])
    sensitive_topics = question_meta.get("sensitive_topics", [])
    
    logger.info(f"Evaluating response with {len(expert_responses)} expert responses")
    
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
    
    # Cultural alignment metrics
    aligned = [c for c in response_cultures if c in relevant_cultures]
    alignment_distribution = Counter(response_cultures)
    cultural_alignment_variance = float(np.var([alignment_distribution[c] for c in relevant_cultures])) if relevant_cultures else 0.0
    
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
        "cultural_alignment_score": len(aligned) / max(1, len(response_cultures)),
        "cultural_alignment_variance": cultural_alignment_variance,
        "unique_cultures": len(set(response_cultures)),
        "diversity_entropy": shannon_entropy(response_cultures) if response_cultures else 0.0,
        "sensitivity_coverage": sensitive_hits / max(1, len(sensitive_topics)) if sensitive_topics else 0,
        "sensitive_topic_mention_rate": sensitive_hits / max(1, len(full_responses)),
        "total_node_latency": 0,  # Would need to track this separately
        "final_response": final_text[:500],  # Truncate for storage
        "final_response_length": final_response_length,
        "final_response_completeness": final_response_completeness,
        "final_sensitivity_coverage": final_sensitive_hits / max(1, len(sensitive_topics)) if sensitive_topics else 0,
        "final_sensitive_topic_mention_rate": final_sensitive_hits / max(1, len(sensitive_topics)) if sensitive_topics else 0,
        "num_full_responses": len(full_responses),
        "num_brief_responses": len(brief_responses)
    }

def evaluate_baseline_response(essay: str) -> dict:
    """Evaluates a baseline essay using LLM-as-judge."""
    length = len(essay)
    completeness = 1.0 if any(opt in essay.lower() for opt in ['a', 'b', 'c', 'd', 'e', 'f']) else 0.0
    
    prompt = f"""You are evaluating an essay for cultural richness. Return only JSON:
{{
  "cultural_alignment_score": <0.0-1.0>,
  "cultural_alignment_variance": <0.0+>,
  "unique_cultures": <0+>,
  "diversity_entropy": <0.0+>,
  "sensitivity_coverage": <0.0-1.0>,
  "sensitive_topic_mention_rate": <0.0-1.0>
}}

Essay: {essay[:1000]}"""
    
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

def compare_with_baseline(n=10):
    """Run comparison between model and baseline."""
    logger.info(f"Starting comparison with n={n} samples")
    
    sampler = PersonaSampler()
    graph = create_smart_cultural_graph()
    model_records, baseline_records = [], []
    
    profiles = sampler.sample_profiles(n)
    
    for i in range(n):
        question, options = sampler.sample_question()
        merged_question = f"{question}\n\nOptions:\n" + "\n".join([
            f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)
        ])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Test {i+1}/{n}")
        logger.info(f"Question: {question}")
        
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
            
            # Add to paired profiles
            paired_profile_metrics.append({
                **profiles[i],
                **model_metrics,
                "sensitivity_score": result['question_meta'].get('sensitivity_score', 0),
                "is_sensitive": result['question_meta'].get('is_sensitive', False)
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
    
    # Simple preprocessing - convert string columns to numeric where possible
    profile_cols = [col for col in df.columns if col not in metric_cols + ['type', 'id', 'question', 'error', 'final_response']]
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    
    # Select numeric columns only
    numeric_df = df[metric_cols].select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Metric Correlations')
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
    
    # Create a summary report
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("Correlation Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total samples analyzed: {len(df)}\n")
        f.write(f"Metrics analyzed: {', '.join(metric_cols)}\n\n")
        
        f.write("Key Findings:\n")
        f.write("-"*30 + "\n")
        
        # Average sensitivity metrics
        if 'sensitivity_score' in df.columns:
            avg_sensitivity = df['sensitivity_score'].mean()
            f.write(f"Average sensitivity score: {avg_sensitivity:.2f}/10\n")
            f.write(f"Sensitive questions: {sum(df['is_sensitive'])}/{len(df)}\n")
        
        if 'num_expert_responses' in df.columns:
            avg_experts = df['num_expert_responses'].mean()
            f.write(f"Average experts consulted: {avg_experts:.1f}\n")
        
        if 'num_full_responses' in df.columns and 'num_brief_responses' in df.columns:
            avg_full = df['num_full_responses'].mean()
            avg_brief = df['num_brief_responses'].mean()
            f.write(f"Average full responses: {avg_full:.1f}\n")
            f.write(f"Average brief responses: {avg_brief:.1f}\n")
    
    logger.info(f"Correlation analysis saved to: {output_dir}")

def create_zip_file(source_dir, output_filename):
    """Create a zip file from a directory."""
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    logger.info(f"Created zip file: {output_filename}")

def main():
    """Run complete analysis pipeline."""
    logger.info("="*60)
    logger.info("CULTURAL ALIGNMENT SYSTEM - COMPLETE RUN")
    logger.info("="*60)
    logger.info("Generating all outputs for 10 test cases")
    
    start_time = time.time()
    
    # Run comparison
    df_results = compare_with_baseline(n=10)
    
    # Save CSV results
    csv_file = save_results_to_csv(df_results)
    
    # Save paired profiles JSON
    json_file = save_paired_profiles_to_json(paired_profile_metrics)
    
    # Run correlation analysis
    analyze_attribute_correlations(paired_profile_metrics)
    
    # Create zip file
    create_zip_file("./correlation_analysis", "correlation_analysis.zip")
    
    # Generate final summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*60)
    logger.info("RUN COMPLETE")
    logger.info("="*60)
    logger.info(f"Total runtime: {total_time:.1f} seconds")
    logger.info("\nGenerated files:")
    logger.info(f"  - {csv_file}")
    logger.info(f"  - {json_file}")
    logger.info(f"  - correlation_analysis.zip")
    logger.info(f"  - run.log")
    
    # Print summary statistics
    model_df = df_results[df_results['type'] == 'model']
    baseline_df = df_results[df_results['type'] == 'baseline']
    
    if not model_df.empty:
        logger.info("\nModel Performance:")
        logger.info(f"  Avg latency: {model_df['latency_seconds'].mean():.1f}s")
        logger.info(f"  Avg response length: {model_df['avg_response_length'].mean():.0f} chars")
        logger.info(f"  Avg cultural alignment: {model_df['cultural_alignment_score'].mean():.2f}")
    
    if not baseline_df.empty:
        logger.info("\nBaseline Performance:")
        logger.info(f"  Avg latency: {baseline_df['latency_seconds'].mean():.1f}s")
        logger.info(f"  Avg response length: {baseline_df['avg_response_length'].mean():.0f} chars")

if __name__ == "__main__":
    main()