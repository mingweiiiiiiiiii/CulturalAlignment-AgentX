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
    """
    Shannon entropy to measure diversity (e.g., of cultural responses).
    """
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0)


def evaluate_response(graph_state: GraphState) -> dict:
    """
    Computes a rich set of evaluation metrics based on graph output.
    Metrics include response quality, alignment, diversity, and topic sensitivity.
    """
    expert_responses = graph_state.get("response_state", {}).get("expert_responses", [])
    relevant_cultures = graph_state.get("question_meta", {}).get("relevant_cultures", [])
    sensitive_topics = graph_state.get("question_meta", {}).get("sensitive_topics", [])

    # Metrics base
    response_lengths = [len(r.get("content", "")) for r in expert_responses]
    response_cultures = [r.get("culture", "") for r in expert_responses if r.get("culture")]

    aligned = [c for c in response_cultures if c in relevant_cultures]
    alignment_distribution = Counter(response_cultures)
    cultural_alignment_variance = float(np.var([alignment_distribution[c] for c in relevant_cultures])) if relevant_cultures else 0.0

    sensitive_hits = sum(
        any(t.lower() in r.get("content", "").lower() for t in sensitive_topics)
        for r in expert_responses
    )

    return {
        "num_expert_responses": len(expert_responses),
        "avg_response_length": sum(response_lengths) / max(1, len(response_lengths)),
        "std_response_length": float(np.std(response_lengths)) if response_lengths else 0.0,
        "response_completeness": sum(
            1 for r in expert_responses if all(opt.lower() in r.get("content", "").lower() for opt in ['a', 'b', 'c', 'd'])
        ) / max(1, len(expert_responses)),
        "cultural_alignment_score": len(aligned) / max(1, len(response_cultures)),
        "cultural_alignment_variance": cultural_alignment_variance,
        "unique_cultures": len(set(response_cultures)),
        "diversity_entropy": shannon_entropy(response_cultures) if response_cultures else 0.0,
        "sensitivity_coverage": sensitive_hits / max(1, len(sensitive_topics)) if sensitive_topics else 0,
        "sensitive_topic_mention_rate": sensitive_hits / max(1, len(expert_responses)),
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
