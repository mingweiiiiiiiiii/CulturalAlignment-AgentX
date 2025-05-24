"""
Modified main.py to use local ollama client for benchmarking.
"""
import json
import math
import os
import time
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from llmagentsetting.ollama_client import OllamaClient
from mylanggraph.graph_optimized import create_optimized_cultural_graph
from utility.baseline import generate_baseline_essay
from utility.inputData import PersonaSampler

# global variable
paired_profile_metrics = []

# Use local ollama client instead of Lambda API
judgeModel = OllamaClient()

def shannon_entropy(labels):
    """Shannon entropy to measure diversity (e.g., of cultural responses)."""
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0)

def evaluate_response(graph_state) -> dict:
    """Computes evaluation metrics based on graph output."""
    expert_responses = graph_state.get("response_state", {}).get("expert_responses", [])
    final_response = graph_state.get("response_state", {}).get("final", "")
    relevant_cultures = graph_state.get("question_meta", {}).get("relevant_cultures", [])
    sensitive_topics = graph_state.get("question_meta", {}).get("sensitive_topics", [])
    
    print(f"Relevant cultures: {relevant_cultures}")
    node_times = graph_state.get("node_times", {})
    print(f"Node times: {node_times}")
    total_node_latency = sum(node_times.values()) if node_times else 0
    
    # Metrics base
    response_lengths = [len(r.get("response", "")) for r in expert_responses]
    response_cultures = [r.get("culture", "") for r in expert_responses if r.get("culture")]
    aligned = [c for c in response_cultures if c in relevant_cultures]
    alignment_distribution = Counter(response_cultures)
    cultural_alignment_variance = float(np.var([alignment_distribution[c] for c in relevant_cultures])) if relevant_cultures else 0.0

    sensitive_hits = sum(
        any(t.lower() in r.get("response", "").lower() for t in sensitive_topics)
        for r in expert_responses
    )
    
    # Metrics for the final response
    final_response_length = len(final_response)
    final_sensitive_hits = sum(t.lower() in final_response.lower() for t in sensitive_topics)
    final_response_completeness = float(
        all(opt.lower() in final_response.lower() for opt in ['a', 'b', 'c', 'd']))

    return {
        "num_expert_responses": len(expert_responses),
        "avg_response_length": sum(response_lengths) / max(1, len(response_lengths)),
        "std_response_length": float(np.std(response_lengths)) if response_lengths else 0.0,
        "response_completeness": sum(
            1 for r in expert_responses if all(opt.lower() in r.get("response", "").lower() for opt in ['a', 'b', 'c', 'd'])
        ) / max(1, len(expert_responses)),
        "cultural_alignment_score": len(aligned) / max(1, len(response_cultures)),
        "cultural_alignment_variance": cultural_alignment_variance,
        "unique_cultures": len(set(response_cultures)),
        "diversity_entropy": shannon_entropy(response_cultures) if response_cultures else 0.0,
        "sensitivity_coverage": sensitive_hits / max(1, len(sensitive_topics)) if sensitive_topics else 0,
        "sensitive_topic_mention_rate": sensitive_hits / max(1, len(expert_responses)),
        "total_node_latency": total_node_latency,
        "final_response": final_response,
        "final_response_length": final_response_length,
        "final_response_completeness": final_response_completeness,
        "final_sensitivity_coverage": final_sensitive_hits / max(1, len(sensitive_topics)),
        "final_sensitive_topic_mention_rate": final_sensitive_hits / (1 if final_response else 1),
    }

def evaluate_baseline_response(essay: str) -> dict:
    """Evaluates a baseline essay using deterministic metrics and LLM-as-judge."""
    # Step 1: Deterministic metrics
    length = len(essay)
    completeness = 1.0 if all(opt in essay.lower() for opt in ['a', 'b', 'c', 'd']) else 0.0

    # Step 2: Construct LLM judging prompt
    prompt = f"""You are an expert evaluator assessing a single essay for cultural richness and sensitivity coverage.
Your task is to assign numeric scores for the following six metrics.

Please return a strict JSON object with these fields:
{{
  "cultural_alignment_score": <float between 0.0 and 1.0>,
  "cultural_alignment_variance": <float >= 0.0>,
  "unique_cultures": <integer >= 0>,
  "diversity_entropy": <float >= 0.0>,
  "sensitivity_coverage": <float between 0.0 and 1.0>,
  "sensitive_topic_mention_rate": <float between 0.0 and 1.0>
}}

Essay to evaluate:
\"\"\"{essay}\"\"\"

Output only the JSON object."""

    # Step 3: Get LLM-generated metric scores
    try:
        response = judgeModel.generate(prompt)
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            judged_metrics = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"LLM response parsing failed: {e}")
        judged_metrics = {
            "cultural_alignment_score": 0.0,
            "cultural_alignment_variance": 0.0,
            "unique_cultures": 0,
            "diversity_entropy": 0.0,
            "sensitivity_coverage": 0.0,
            "sensitive_topic_mention_rate": 0.0,
        }

    # Step 4: Merge and return full evaluation
    return {
        "num_expert_responses": 1,
        "avg_response_length": length,
        "std_response_length": 0.0,
        "response_completeness": completeness,
        **judged_metrics
    }

def compare_with_baseline(n=3):  # Reduced to 3 for testing
    sampler = PersonaSampler()
    graph = create_optimized_cultural_graph()
    model_records, baseline_records = [], []

    profiles = sampler.sample_profiles(n)
    for i in range(n):
        question, options = sampler.sample_question()
        merged_question = f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])

        print(f"\n{'='*80}")
        print(f"BENCHMARK RUN {i+1}/{n}")
        print(f"{'='*80}")
        print(f"Question: {merged_question}")

        # --- Model system ---
        state = {
            "user_profile": profiles[i],
            "question_meta": {
                "original": merged_question,
                "options": options,
                "sensitive_topics": [],
                "relevant_cultures": [],
            },
            "response_state": {
                "expert_responses": [],
                "final": ""
            },
            "is_sensitive": False,
            "activate_sensitivity_check": True,
            "activate_extract_topics": False,
            "activate_router": False,
            "activate_compose": False,
            "current_state": "sensitivity_check",
            "__start__": "sensitivity_check",
            "steps": [],
            "expert_consultations": {},
            "final_response": {},
            "planner_counter": 0,
            "node_times": {}
        }
        
        print(f"\nRunning cultural agent model...")
        model_start = time.perf_counter()
        
        try:
            result = graph.invoke(state, config={
                "configurable": {"thread_id": f"benchmark_{i}"},
            })
            model_end = time.perf_counter()
            model_latency = model_end - model_start

            model_metrics = evaluate_response(result)
            model_metrics.update({"type": "model", "id": i, "latency_seconds": model_latency})
            model_records.append(model_metrics)
            print(f"✅ Model completed (Latency: {model_latency:.3f}s)")
            
            paired_profile_metrics.append({**profiles[i], **model_metrics})
            
        except Exception as e:
            print(f"❌ Model failed: {e}")
            model_metrics = {
                "type": "model", "id": i, "latency_seconds": 0,
                "num_expert_responses": 0, "avg_response_length": 0,
                "response_completeness": 0, "cultural_alignment_score": 0,
                "diversity_entropy": 0, "sensitivity_coverage": 0,
                "sensitive_topic_mention_rate": 0, "final_response": "ERROR"
            }
            model_records.append(model_metrics)

        # --- Baseline ---
        print(f"\nRunning baseline...")
        baseline_start = time.perf_counter()
        
        try:
            essay = generate_baseline_essay(profiles, merged_question)
            baseline_end = time.perf_counter()
            baseline_latency = baseline_end - baseline_start
            
            baseline_metrics = evaluate_baseline_response(essay)
            baseline_metrics.update({"type": "baseline", "id": i, "latency_seconds": baseline_latency})
            baseline_records.append(baseline_metrics)
            print(f"✅ Baseline completed (Latency: {baseline_latency:.3f}s)")
            
        except Exception as e:
            print(f"❌ Baseline failed: {e}")
            baseline_metrics = {
                "type": "baseline", "id": i, "latency_seconds": 0,
                "num_expert_responses": 1, "avg_response_length": 0,
                "response_completeness": 0, "cultural_alignment_score": 0,
                "diversity_entropy": 0, "sensitivity_coverage": 0,
                "sensitive_topic_mention_rate": 0
            }
            baseline_records.append(baseline_metrics)

    return pd.DataFrame(model_records + baseline_records)

def generate_comparison_table_markdown(df: pd.DataFrame) -> str:
    metrics = [
        "avg_response_length", "response_completeness", "cultural_alignment_score",
        "diversity_entropy", "sensitivity_coverage", "sensitive_topic_mention_rate", "latency_seconds"
    ]
    table = "| Metric | Baseline Avg | Model Avg |\n|--------|---------------|-----------|\n"
    for metric in metrics:
        baseline_avg = df[df["type"] == "baseline"][metric].mean()
        model_avg = df[df["type"] == "model"][metric].mean()
        table += f"| {metric} | {baseline_avg:.3f} | {model_avg:.3f} |\n"
    return table

def save_results(df: pd.DataFrame, paired_metrics: list):
    """Save all results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    os.makedirs("./results", exist_ok=True)
    
    # Save DataFrame
    csv_path = f"./results/benchmark_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results CSV saved to: {csv_path}")
    
    # Save paired metrics JSON
    json_path = f"./results/paired_metrics_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(paired_metrics, f, indent=2)
    print(f"✅ Paired metrics JSON saved to: {json_path}")
    
    # Save markdown comparison
    md_content = f"# Benchmark Results - {timestamp}\n\n"
    md_content += f"## Summary\n\n"
    md_content += f"- Total runs: {len(df) // 2}\n"
    md_content += f"- Model: Optimized Cultural Agent (granite3.3)\n"
    md_content += f"- Baseline: Standard Essay Generator\n\n"
    md_content += f"## Comparison Table\n\n"
    md_content += generate_comparison_table_markdown(df)
    
    md_path = f"./results/benchmark_summary_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"✅ Summary markdown saved to: {md_path}")
    
    return csv_path, json_path, md_path

if __name__ == "__main__":
    print("Starting Cultural Agent Benchmark (Local Ollama)")
    print("=" * 80)
    
    # Run benchmark
    df_results = compare_with_baseline(n=3)  # Run 3 tests
    
    # Save results
    csv_path, json_path, md_path = save_results(df_results, paired_profile_metrics)
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Summary: {md_path}")