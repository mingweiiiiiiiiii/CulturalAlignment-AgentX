import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys
sys.path.append('.')  # Adjust as needed

from mylanggraph.graph import create_cultural_graph
from mylanggraph.types import GraphState
from utility.inputData import PersonaSampler
from utility.baseline import generate_baseline_essay


def shannon_entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0)


def evaluate_response_from_model(state: GraphState) -> dict:
    responses = state.get("response_state", {}).get("expert_responses", [])
    lengths = [len(r.get("content", "")) for r in responses]
    cultures = [r.get("culture", "") for r in responses if r.get("culture")]

    return {
        "num_expert_responses": len(responses),
        "avg_response_length": np.mean(lengths) if lengths else 0,
        "std_response_length": np.std(lengths) if lengths else 0,
        "response_completeness": sum(
            1 for r in responses if all(x in r.get("content", "").lower() for x in ['a', 'b', 'c', 'd'])
        ) / max(1, len(responses)),
        "cultural_alignment_score": 0.0,  # No relevant_cultures provided
        "cultural_alignment_variance": 0.0,
        "unique_cultures": len(set(cultures)),
        "diversity_entropy": shannon_entropy(cultures) if cultures else 0.0,
        "sensitivity_coverage": 0.0,
        "sensitive_topic_mention_rate": 0.0,
    }


def evaluate_baseline_response(essay: str) -> dict:
    length = len(essay)
    return {
        "num_expert_responses": 1,
        "avg_response_length": length,
        "std_response_length": 0.0,
        "response_completeness": int(all(opt in essay.lower() for opt in ['a', 'b', 'c', 'd'])),
        "cultural_alignment_score": 0.0,
        "cultural_alignment_variance": 0.0,
        "unique_cultures": 0,
        "diversity_entropy": 0.0,
        "sensitivity_coverage": 0.0,
        "sensitive_topic_mention_rate": 0.0,
    }


def compare_with_baseline(n=10):
    sampler = PersonaSampler()
    graph = create_cultural_graph()
    model_records, baseline_records = [], []

    for i in range(n):
        profiles = sampler.sample_profiles(n=1)
        question, options = sampler.sample_question()
        merged_question = f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])

        # --- Model system ---
        state: GraphState = {
            "user_profile": profiles[0],
            "question_meta": {
                "original": merged_question,
                "options": options,
                "sensitive_topics": [],
                "relevant_cultures": [],
            },
            "response_state": {
                "expert_responses": [],
            },
            "full_history": [],
            "planner_counter": 0,
            "activate_sensitivity_check": True,
            "activate_extract_topics": True,
            "activate_router": False,
            "activate_judge": False,
            "activate_compose": False,
            "current_state": "planner",
        }
    
        result = graph.invoke(state, config={
            "recursion_limit": 200,
            "configurable": {"thread_id": str(i)},
            "verbose": True,
        })      
        model_metrics = evaluate_response_from_model(result)
        model_metrics.update({"type": "model", "id": i})
        model_records.append(model_metrics)

        # --- Baseline ---
        essay = generate_baseline_essay(profiles, merged_question)
        baseline_metrics = evaluate_baseline_response(essay)
        baseline_metrics.update({"type": "baseline", "id": i})
        baseline_records.append(baseline_metrics)

    return pd.DataFrame(model_records + baseline_records)


def generate_comparison_table_markdown(df: pd.DataFrame) -> str:
    metrics = [
        "avg_response_length", "response_completeness", "cultural_alignment_score",
        "diversity_entropy", "sensitivity_coverage", "sensitive_topic_mention_rate"
    ]
    table = "| Metric | Baseline Avg | Model Avg |\n|--------|---------------|-----------|\n"
    for metric in metrics:
        baseline_avg = df[df["type"] == "baseline"][metric].mean()
        model_avg = df[df["type"] == "model"][metric].mean()
        table += f"| {metric} | {baseline_avg:.3f} | {model_avg:.3f} |\n"
    return table


def save_markdown_table(df: pd.DataFrame, path: str = "./comparison_table.md"):
    markdown = generate_comparison_table_markdown(df)
    with open(path, "w") as f:
        f.write("# Baseline vs Model Comparison Table\n\n")
        f.write(markdown)
    return path




if __name__ == "__main__":
    df_results = compare_with_baseline(n=5)
    path = save_markdown_table(df_results)
    print(f"\n✅ Markdown saved to: {path}")
