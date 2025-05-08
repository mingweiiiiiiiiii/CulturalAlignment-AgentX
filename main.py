import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys
sys.path.append('.')  # Adjust as needed
import json
from mylanggraph.graph import create_cultural_graph
from mylanggraph.types import GraphState
from utility.inputData import PersonaSampler
from utility.baseline import generate_baseline_essay

from llmagentsetting import llm_clients
use_gemini = False
if use_gemini:
    judgeModel = llm_clients.GeminiClient()
else:
    print("❗Falling back to Lambda API client")
    judgeModel = llm_clients.LambdaAPIClient()


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
    final_response = graph_state.get("response_state", {}).get("final", "")
    relevant_cultures = graph_state.get("question_meta", {}).get("relevant_cultures", [])
    sensitive_topics = graph_state.get("question_meta", {}).get("sensitive_topics", [])
    question = graph_state.get("question_meta", {}).get("original", "")
    # Add these to visualize responses, free feel to comment it out
    print(f"question: {question}")
    print(f"Final response: {final_response}")
    print(f"Evaluating {len(expert_responses)} expert responses...")
    print(f"Relevant cultures: {relevant_cultures}")
    print(f"Sensitive topics: {sensitive_topics}")
    
    # Metrics base
    response_lengths = [len(r.get("response", "")) for r in expert_responses]
    response_cultures = [r.get("culture", "") for r in expert_responses if r.get("culture")]
    print(f"Response cultures: {response_cultures} len: {len(response_cultures)}")
    aligned = [c for c in response_cultures if c in relevant_cultures]
    alignment_distribution = Counter(response_cultures)
    print(f"Alignment distribution: {alignment_distribution}")
    cultural_alignment_variance = float(np.var([alignment_distribution[c] for c in relevant_cultures])) if relevant_cultures else 0.0

    sensitive_hits = sum(
        any(t.lower() in r.get("response", "").lower() for t in sensitive_topics)
        for r in expert_responses
    )

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
    }

# LLM-AS-JUDGE
def evaluate_baseline_response(essay: str) -> dict:
    """
    Evaluates a baseline essay using deterministic metrics and LLM-as-a-judge
    for cultural alignment, diversity, and sensitivity metrics.
    """
    # Step 1: Deterministic metrics
    length = len(essay)
    completeness = 1.0 if all(opt in essay.lower() for opt in ['a', 'b', 'c', 'd']) else 0.0

    # Step 2: Construct LLM judging prompt
    prompt = f"""
You are an expert evaluator assessing a single essay for cultural richness and sensitivity coverage.
Your task is to assign **numeric scores** for the following six metrics, based solely on the essay content.

Please return a **strict JSON object** with these fields and formats (no explanation, no extra text):

{{
  "cultural_alignment_score": float between 0.0 and 1.0,
  "cultural_alignment_variance": float ≥ 0.0,
  "unique_cultures": integer ≥ 0,
  "diversity_entropy": float ≥ 0.0,
  "sensitivity_coverage": float between 0.0 and 1.0,
  "sensitive_topic_mention_rate": float between 0.0 and 1.0
}}

### Metric Guidance:
- **cultural_alignment_score**: How well the essay reflects recognizable cultural perspectives (e.g., values, customs, norms). 1.0 = highly culturally grounded; 0.0 = culturally generic or monocultural.
- **cultural_alignment_variance**: Variation in emphasis across multiple cultures. High variance = one culture dominates; low variance = more balanced.
- **unique_cultures**: Number of clearly distinct cultural perspectives or references included (e.g., references to Japanese collectivism, Western individualism, Islamic ethics).
- **diversity_entropy**: Shannon entropy of cultural variety — more balance and uniqueness → higher score.
- **sensitivity_coverage**: Whether the essay explicitly addresses socially sensitive issues (e.g., gender, inequality, religion, race). Higher = better coverage.
- **sensitive_topic_mention_rate**: How frequently these sensitive topics are mentioned. Do not penalize shortness; this is a ratio of mentions to expected topics.

### Essay to evaluate:
\"\"\"{essay}\"\"\"

Output only the JSON object with the six keys. No prose.
"""

    # Step 3: Get LLM-generated metric scores
    try:
        if use_gemini:
            generate_response = judgeModel.generate(prompt)
        elif isinstance(judgeModel, llm_clients.LambdaAPIClient):
            generate_response = judgeModel.get_completion(
            prompt,
            temperature=0,
            max_tokens=200
        )
            print("RAW LLM OUTPUT ▶", repr(generate_response))
        else:
            raise RuntimeError("LLM client does not have a supported generate method")
        
        judged_metrics = json.loads(generate_response) if isinstance(generate_response, str) else generate_response
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


def compare_with_baseline(n=10):
    sampler = PersonaSampler()
    graph = create_cultural_graph()
    model_records, baseline_records = [], []

    profiles = sampler.sample_profiles(n)
    for i in range(n):
        question, options = sampler.sample_question()
        merged_question = f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])

        # --- Model system ---
        state: GraphState = {
            "user_profile": profiles[i],
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
        print(f"\n\n--- Model {i} ---")
        print("Graph nodes:", graph.get_graph().nodes)
        print("\nInvoking graph with state:", state)
        result = graph.invoke(state, config={
            "recursion_limit": 200,
            "configurable": {"thread_id": str(i)},
            "verbose": True,
        })     
        model_metrics = evaluate_response(result)
        model_metrics.update({"type": "model", "id": i})
        model_records.append(model_metrics)
        print("Model metrics:", model_metrics)
        # --- Baseline ---
        essay = generate_baseline_essay(profiles, merged_question)
        print(f"Baseline LLM response {i}: {essay}")
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
    df_results = compare_with_baseline(n=3)  # Adjust n as needed
    path = save_markdown_table(df_results)
    print(f"\n✅ Markdown saved to: {path}")
