import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Fix path issue if modules aren't found
import sys
sys.path.append('.')  # Adjust as needed

from mylanggraph.graph import create_cultural_graph        # Main workflow definition
from utility.inputData import PersonaSampler                # For user profile/question sampling
from mylanggraph.types import GraphState                    # Graph state interface


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


def run_experiments(n=100):
    """
    Runs LangGraph for n different simulated personas and logs results.
    """
    sampler = PersonaSampler()
    graph = create_cultural_graph()
    records = []

    for i in range(n):
        profiles = sampler.sample_profiles(n=1)
        question, options = sampler.sample_question()
        options_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
        merged_question = f"{question}\n\nOptions:\n{options_str}"

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

        metrics = evaluate_response(result)
        records.append({**profiles[0], **metrics})

    return pd.DataFrame(records)


def visualize_metrics(df: pd.DataFrame):
    """
    Visualizes evaluation metrics via boxplots, bar charts, and a correlation heatmap.
    """
    plt.figure(figsize=(10, 5))
    df.boxplot(column="cultural_alignment_score", by="religion")
    plt.title("Cultural Alignment Score by Religion")
    plt.suptitle("")
    plt.xlabel("Religion")
    plt.ylabel("Alignment Score")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    df.groupby("education")["avg_response_length"].mean().plot(kind='bar')
    plt.title("Avg Response Length by Education")
    plt.ylabel("Length")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    corr = df[[
        "num_expert_responses", "avg_response_length", "std_response_length",
        "response_completeness", "cultural_alignment_score", "cultural_alignment_variance",
        "unique_cultures", "diversity_entropy", "sensitivity_coverage", "sensitive_topic_mention_rate"
    ]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Evaluation Metrics")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_results = run_experiments(n=100)  
    visualize_metrics(df_results)
