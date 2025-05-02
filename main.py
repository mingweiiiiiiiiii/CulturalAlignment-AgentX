import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.optimize import minimize

# Fix path issue if modules aren't found
import sys
sys.path.append('.')  # Adjust as needed

from mylanggraph.graph import create_cultural_graph        # Main workflow definition
from utility.inputData import PersonaSampler                # For user profile/question sampling
from mylanggraph.types import GraphState                    # Graph state interface

def shannon_entropy(labels):
    """
    Calculate Shannon entropy of a list of labels to measure diversity.
    """
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0)

def multi_objective_loss(x, weights):
    """
    Multi-objective loss function used for Pareto optimization.
    Converts a maximization problem into a minimization one.
    """
    neg_x = [-val for val in x]  # Negate to minimize
    return sum(w * v for w, v in zip(weights, neg_x))

def infer_weights_from_correlation(df: pd.DataFrame):
    """
    Infer metric weights based on correlation matrix.
    Less correlated metrics are prioritized to maintain metric diversity.
    """
    metrics = [
        "cultural_alignment_score",
        "diversity_entropy",
        "sensitive_topic_mention_rate",
        "response_completeness"
    ]
    corr_matrix = df[metrics].corr().abs()
    avg_corr = corr_matrix.mean()  # Compute mean correlation per metric
    weights = 1.0 - avg_corr        # Invert to give higher weight to less correlated
    weights /= weights.sum()       # Normalize to sum to 1
    return weights.tolist()

def evaluate_response(graph_state: GraphState, weights=None) -> dict:
    """
    Evaluate key metrics from graph state and apply multi-objective optimization if weights are provided.
    """
    expert_responses = graph_state.get("response_state", {}).get("expert_responses", [])
    relevant_cultures = graph_state.get("question_meta", {}).get("relevant_cultures", [])
    sensitive_topics = graph_state.get("question_meta", {}).get("sensitive_topics", [])

    # Extract basic stats from responses
    response_lengths = [len(r.get("content", "")) for r in expert_responses]
    response_cultures = [r.get("culture", "") for r in expert_responses if r.get("culture")]

    aligned = [c for c in response_cultures if c in relevant_cultures]
    alignment_distribution = Counter(response_cultures)
    cultural_alignment_variance = float(np.var([alignment_distribution[c] for c in relevant_cultures])) if relevant_cultures else 0.0

    # Count how many sensitive topics appear in responses
    sensitive_hits = sum(
        any(t.lower() in r.get("content", "").lower() for t in sensitive_topics)
        for r in expert_responses
    )

    # Core evaluation metrics
    cultural_alignment_score = len(aligned) / max(1, len(response_cultures))
    diversity_entropy = shannon_entropy(response_cultures) if response_cultures else 0.0
    sensitive_topic_mention_rate = sensitive_hits / max(1, len(expert_responses))
    response_completeness = sum(
        1 for r in expert_responses if all(opt.lower() in r.get("content", "").lower() for opt in ['a', 'b', 'c', 'd'])
    ) / max(1, len(expert_responses))

    metrics_array = [cultural_alignment_score, diversity_entropy, sensitive_topic_mention_rate, response_completeness]

    # Optimize only if weights are supplied
    if weights:
        res = minimize(lambda x: multi_objective_loss(x, weights), metrics_array, method='SLSQP', bounds=[(0,1)]*4)
        pareto_score = -res.fun  # Convert back to maximization score
    else:
        pareto_score = np.mean(metrics_array)  # fallback to average if weights unavailable

    # Return all metrics
    return {
        "num_expert_responses": len(expert_responses),
        "avg_response_length": sum(response_lengths) / max(1, len(response_lengths)),
        "std_response_length": float(np.std(response_lengths)) if response_lengths else 0.0,
        "response_completeness": response_completeness,
        "cultural_alignment_score": cultural_alignment_score,
        "cultural_alignment_variance": cultural_alignment_variance,
        "unique_cultures": len(set(response_cultures)),
        "diversity_entropy": diversity_entropy,
        "sensitivity_coverage": sensitive_hits / max(1, len(sensitive_topics)) if sensitive_topics else 0,
        "sensitive_topic_mention_rate": sensitive_topic_mention_rate,
        "pareto_optimal_score": pareto_score
    }

def run_experiments(n=100):
    """
    Run LangGraph over 'n' profiles.
    First, do a bootstrap run to compute metric correlations and infer weights.
    Then, re-evaluate with multi-objective optimization.
    """
    sampler = PersonaSampler()
    graph = create_cultural_graph()
    records = []

    # Bootstrap to estimate weights from metric correlation
    bootstrap = []
    for _ in range(10):
        profiles = sampler.sample_profiles(n=1)
        question, options = sampler.sample_question()
        options_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
        merged_question = f"{question}\n\nOptions:\n{options_str}"

        state = {
            "user_profile": profiles,
            "question_meta": {
                "original": merged_question,
                "options": options,
                "sensitive_topics": [],
                "relevant_cultures": [],
            },
            "response_state": {"expert_responses": []},
            "full_history": [],
            "planner_counter": 0,
            "activate_sensitivity_check": True,
            "activate_extract_topics": True,
            "activate_router": False,
            "activate_judge": False,
            "activate_compose": False,
            "current_state": "planner",
        }
        result = graph.invoke(state, config={"recursion_limit": 200, "configurable": {"thread_id": str(_)}})
        bootstrap.append(evaluate_response(result, weights=None))

    # Derive dynamic weights for optimization
    weights = infer_weights_from_correlation(pd.DataFrame(bootstrap))
    print("Inferred Weights from Correlation:", weights)

    # Main experiment loop
    for i in range(n):
        profiles = sampler.sample_profiles(n=1)
        question, options = sampler.sample_question()
        options_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
        merged_question = f"{question}\n\nOptions:\n{options_str}"

        state: GraphState = {
            "user_profile": profiles,
            "question_meta": {
                "original": merged_question,
                "options": options,
                "sensitive_topics": [],
                "relevant_cultures": [],
            },
            "response_state": {"expert_responses": []},
            "full_history": [],
            "planner_counter": 0,
            "activate_sensitivity_check": True,
            "activate_extract_topics": True,
            "activate_router": False,
            "activate_judge": False,
            "activate_compose": False,
            "current_state": "planner",
        }
        result = graph.invoke(state, config={"recursion_limit": 200, "configurable": {"thread_id": str(i)}})
        metrics = evaluate_response(result, weights)
        records.append({**profiles[0], **metrics})

    return pd.DataFrame(records)

def analyze_metric_relationships(df: pd.DataFrame):
    """
    Visualizes correlation between key evaluation metrics to detect trade-offs or synergies.
    """
    metrics = [
        "cultural_alignment_score",
        "diversity_entropy",
        "sensitive_topic_mention_rate",
        "response_completeness"
    ]
    corr_matrix = df[metrics].corr()
    print("Metric Correlation Matrix:")
    print(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation of Key Metrics (Pre-Optimization)")
    plt.tight_layout()
    plt.show()

def visualize_metrics(df: pd.DataFrame):
    """
    Generate visual summaries of metric distributions and interrelations.
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
        "unique_cultures", "diversity_entropy", "sensitivity_coverage", "sensitive_topic_mention_rate",
        "pareto_optimal_score"
    ]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Evaluation Metrics")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run experiments, then analyze interdependencies and show final metric visualizations
    df_results = run_experiments(n=100)
    analyze_metric_relationships(df_results)
    visualize_metrics(df_results)