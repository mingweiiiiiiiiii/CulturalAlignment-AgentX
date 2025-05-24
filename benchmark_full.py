"""
Full benchmark comparing optimized cultural agent with baseline.
Uses local ollama throughout.
"""
import sys
sys.path.append('/app')

import json
import time
import os
from datetime import datetime
from collections import Counter
import math

from llmagentsetting.ollama_client import OllamaClient
from node.sensitivity_optimized import analyze_question_sensitivity
from utility.inputData import PersonaSampler
import numpy as np
import pandas as pd

# Initialize components
client = OllamaClient()
sampler = PersonaSampler()

def get_cultural_expert_response(culture: str, question: str) -> str:
    """Get response from a cultural expert using local ollama."""
    prompt = f"""You are a cultural expert from {culture}, deeply familiar with its values and perspectives.
Answer this question from your cultural viewpoint in under 150 words:

Question: {question}"""
    
    try:
        return client.generate(prompt)
    except Exception as e:
        return f"Error: {str(e)}"

def compose_final_response(question: str, expert_responses: dict, user_profile: dict) -> str:
    """Compose final culturally-aligned response."""
    expert_summaries = "\n\n".join([
        f"{culture}: {response[:200]}..."
        for culture, response in expert_responses.items()
    ])
    
    prompt = f"""Synthesize these cultural perspectives into a balanced response:

Question: {question}

User is from {user_profile.get('location', 'Unknown')} with {user_profile.get('cultural_background', 'Unknown')} background.

Expert perspectives:
{expert_summaries}

Provide a balanced response (under 200 words) that considers multiple viewpoints:"""
    
    try:
        return client.generate(prompt)
    except Exception as e:
        return f"Error composing response: {str(e)}"

def generate_baseline_response(question: str, profile: dict) -> str:
    """Generate baseline response without cultural alignment."""
    prompt = f"""Answer this question thoughtfully:

Question: {question}

Provide a balanced response considering general perspectives:"""
    
    try:
        return client.generate(prompt)
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_response_metrics(response: str, expert_responses: list = None) -> dict:
    """Calculate response metrics."""
    metrics = {
        "response_length": len(response),
        "response_completeness": 1.0 if any(opt in response.lower() for opt in ['a', 'b', 'c', 'd', 'e']) else 0.0,
    }
    
    if expert_responses:
        metrics["num_experts"] = len(expert_responses)
        metrics["avg_expert_length"] = sum(len(r) for r in expert_responses) / len(expert_responses) if expert_responses else 0
    
    return metrics

def run_cultural_agent_test(question: str, profile: dict) -> dict:
    """Run the optimized cultural agent workflow."""
    start_time = time.time()
    
    # Step 1: Sensitivity Analysis
    state = {
        "question_meta": {"original": question},
        "user_profile": profile
    }
    
    sensitivity_result = analyze_question_sensitivity(state)
    meta = sensitivity_result["question_meta"]
    
    # Step 2: Cultural Expert Consultation (if sensitive)
    expert_responses = {}
    if meta['is_sensitive']:
        cultures = ["United States", "Japan", "India"]  # Fixed set for consistency
        for culture in cultures:
            expert_responses[culture] = get_cultural_expert_response(culture, question)
    
    # Step 3: Compose Final Response
    if expert_responses:
        final_response = compose_final_response(question, expert_responses, profile)
    else:
        final_response = generate_baseline_response(question, profile)
    
    end_time = time.time()
    
    # Calculate metrics
    metrics = evaluate_response_metrics(final_response, list(expert_responses.values()) if expert_responses else None)
    metrics.update({
        "latency": end_time - start_time,
        "is_sensitive": meta['is_sensitive'],
        "sensitivity_score": meta['sensitivity_score'],
        "num_cultures_consulted": len(expert_responses),
        "final_response": final_response[:500]  # Truncate for storage
    })
    
    return metrics

def run_baseline_test(question: str, profile: dict) -> dict:
    """Run baseline test."""
    start_time = time.time()
    
    response = generate_baseline_response(question, profile)
    
    end_time = time.time()
    
    metrics = evaluate_response_metrics(response)
    metrics.update({
        "latency": end_time - start_time,
        "is_sensitive": False,
        "sensitivity_score": 0,
        "num_cultures_consulted": 0,
        "final_response": response[:500]
    })
    
    return metrics

def run_benchmark(n_tests: int = 5):
    """Run full benchmark comparison."""
    print("=" * 80)
    print(f"CULTURAL ALIGNMENT BENCHMARK - {n_tests} Tests")
    print("=" * 80)
    print(f"Model: Optimized Cultural Agent (granite3.3:latest)")
    print(f"Baseline: Simple Response Generator")
    print("-" * 80)
    
    results = []
    
    # Sample profiles and questions
    profiles = sampler.sample_profiles(n_tests)
    
    for i in range(n_tests):
        question, options = sampler.sample_question()
        merged_question = f"{question}\n\nOptions:\n" + "\n".join([
            f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)
        ])
        
        print(f"\n[Test {i+1}/{n_tests}]")
        print(f"Question: {question[:80]}...")
        
        # Run cultural agent
        print("  Running cultural agent...", end='', flush=True)
        try:
            agent_metrics = run_cultural_agent_test(merged_question, profiles[i])
            agent_metrics["type"] = "cultural_agent"
            agent_metrics["test_id"] = i
            agent_metrics["question"] = question[:100]
            results.append(agent_metrics)
            print(f" ✓ ({agent_metrics['latency']:.1f}s, sensitive={agent_metrics['is_sensitive']})")
        except Exception as e:
            print(f" ✗ Error: {e}")
            results.append({
                "type": "cultural_agent",
                "test_id": i,
                "error": str(e),
                "latency": 0
            })
        
        # Run baseline
        print("  Running baseline...", end='', flush=True)
        try:
            baseline_metrics = run_baseline_test(merged_question, profiles[i])
            baseline_metrics["type"] = "baseline"
            baseline_metrics["test_id"] = i
            baseline_metrics["question"] = question[:100]
            results.append(baseline_metrics)
            print(f" ✓ ({baseline_metrics['latency']:.1f}s)")
        except Exception as e:
            print(f" ✗ Error: {e}")
            results.append({
                "type": "baseline",
                "test_id": i,
                "error": str(e),
                "latency": 0
            })
    
    return pd.DataFrame(results)

def generate_summary_report(df: pd.DataFrame) -> str:
    """Generate markdown summary report."""
    agent_df = df[df['type'] == 'cultural_agent']
    baseline_df = df[df['type'] == 'baseline']
    
    report = f"""# Cultural Alignment Benchmark Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

| Metric | Cultural Agent | Baseline |
|--------|----------------|----------|
| Avg Latency (s) | {agent_df['latency'].mean():.2f} | {baseline_df['latency'].mean():.2f} |
| Avg Response Length | {agent_df['response_length'].mean():.0f} | {baseline_df['response_length'].mean():.0f} |
| Sensitivity Detection Rate | {(agent_df['is_sensitive'].sum() / len(agent_df) * 100):.1f}% | N/A |
| Avg Sensitivity Score | {agent_df['sensitivity_score'].mean():.1f}/10 | N/A |
| Cultures Consulted (when sensitive) | {agent_df[agent_df['is_sensitive']]['num_cultures_consulted'].mean():.1f} | 0 |

## Test Details

"""
    
    # Add individual test results
    for i in range(len(agent_df)):
        agent_row = agent_df[agent_df['test_id'] == i].iloc[0]
        baseline_row = baseline_df[baseline_df['test_id'] == i].iloc[0]
        
        report += f"""### Test {i+1}
**Question**: {agent_row.get('question', 'N/A')}

**Cultural Agent**:
- Sensitivity: {'Yes' if agent_row.get('is_sensitive', False) else 'No'} (Score: {agent_row.get('sensitivity_score', 0)}/10)
- Cultures Consulted: {agent_row.get('num_cultures_consulted', 0)}
- Response Length: {agent_row.get('response_length', 0)} chars
- Latency: {agent_row.get('latency', 0):.2f}s

**Baseline**:
- Response Length: {baseline_row.get('response_length', 0)} chars
- Latency: {baseline_row.get('latency', 0):.2f}s

---

"""
    
    return report

def main():
    """Run the benchmark and save results."""
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Run benchmark
    print("\nStarting benchmark...")
    df = run_benchmark(n_tests=5)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw results
    csv_path = f'./results/benchmark_raw_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Raw results saved to: {csv_path}")
    
    # Generate and save summary report
    report = generate_summary_report(df)
    report_path = f'./results/benchmark_report_{timestamp}.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Summary report saved to: {report_path}")
    
    # Save to host directory as well
    host_csv = f'/app/benchmark_results_{timestamp}.csv'
    host_report = f'/app/benchmark_report_{timestamp}.md'
    df.to_csv(host_csv, index=False)
    with open(host_report, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Results also saved to container root:")
    print(f"   - {host_csv}")
    print(f"   - {host_report}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    agent_avg_latency = df[df['type'] == 'cultural_agent']['latency'].mean()
    baseline_avg_latency = df[df['type'] == 'baseline']['latency'].mean()
    print(f"Cultural Agent avg latency: {agent_avg_latency:.2f}s")
    print(f"Baseline avg latency: {baseline_avg_latency:.2f}s")
    print(f"Speedup: {baseline_avg_latency/agent_avg_latency:.2f}x")

if __name__ == "__main__":
    main()