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

from llmagentsetting import llm_clients
from mylanggraph.graph import create_cultural_graph
from mylanggraph.types import GraphState
from utility.baseline import generate_baseline_essay
from utility.inputData import PersonaSampler

# global variable
paired_profile_metrics = []
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
    expert_responses = graph_state.get(
        "response_state", {}).get("expert_responses", [])
    final_response = graph_state.get("response_state", {}).get("final", "")
    relevant_cultures = graph_state.get(
        "question_meta", {}).get("relevant_cultures", [])
    sensitive_topics = graph_state.get(
        "question_meta", {}).get("sensitive_topics", [])
    question = graph_state.get("question_meta", {}).get("original", "")
    # Add these to visualize responses, free feel to comment it out
    # print(f"question: {question}")
    # print(f"Final response: {final_response}")
    # print(f"Evaluating {len(expert_responses)} expert responses...")
    print(f"Relevant cultures: {relevant_cultures}")
    # print(f"Sensitive topics: {sensitive_topics}")
    node_times = graph_state.get("node_times", {})
    print(f"Node times: {node_times}")
    total_node_latency = sum(node_times.values())
    # Metrics base
    response_lengths = [len(r.get("response", "")) for r in expert_responses]
    response_cultures = [r.get("culture", "")
                         for r in expert_responses if r.get("culture")]
    print(
        f"Response cultures: {response_cultures} len: {len(response_cultures)}")
    aligned = [c for c in response_cultures if c in relevant_cultures]
    alignment_distribution = Counter(response_cultures)
    print(f"Alignment distribution: {alignment_distribution}")
    cultural_alignment_variance = float(np.var(
        [alignment_distribution[c] for c in relevant_cultures])) if relevant_cultures else 0.0

    sensitive_hits = sum(
        any(t.lower() in r.get("response", "").lower()
            for t in sensitive_topics)
        for r in expert_responses
    )
    # Metrics for the final response
    final_response_length = len(final_response)
    final_sensitive_hits = sum(t.lower() in final_response.lower()
                               for t in sensitive_topics)
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
        # Final composed response
        "total_node_latency": total_node_latency,
        "final_response": final_response,
        "final_response_length": final_response_length,
        "final_response_completeness": final_response_completeness,
        "final_sensitivity_coverage": final_sensitive_hits / max(1, len(sensitive_topics)),
        "final_sensitive_topic_mention_rate": final_sensitive_hits / (1 if final_response else 1),
    }

# LLM-AS-JUDGE


def evaluate_baseline_response(essay: str) -> dict:
    """
    Evaluates a baseline essay using deterministic metrics and LLM-as-a-judge
    for cultural alignment, diversity, and sensitivity metrics.
    """
    # Step 1: Deterministic metrics
    length = len(essay)
    completeness = 1.0 if all(opt in essay.lower()
                              for opt in ['a', 'b', 'c', 'd']) else 0.0

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
            raise RuntimeError(
                "LLM client does not have a supported generate method")

        judged_metrics = json.loads(generate_response) if isinstance(
            generate_response, str) else generate_response
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
        merged_question = f"{question}\n\nOptions:\n" + \
            "\n".join([f"{chr(65 + j)}. {opt}" for j,
                      opt in enumerate(options)])

        print(f"Merged question: {merged_question}")

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
        # print("Graph nodes:", graph.get_graph().nodes)
        # print("\nInvoking graph with state:", state)
        model_start = time.perf_counter()
        result = graph.invoke(state, config={
            "recursion_limit": 200,
            "configurable": {"thread_id": str(i)},
            "verbose": True,
        })
        model_end = time.perf_counter()
        model_latency = model_end - model_start
        print("API Calls per Node:")
        for node, count in result.get("api_calls", {}).items():
            print(f"  {node}: {count} calls")

        model_metrics = evaluate_response(result)
        model_metrics.update(
            {"type": "model", "id": i, "latency_seconds": model_latency})
        model_records.append(model_metrics)
        print(f"Model metrics (Latency: {model_latency:.3f}s):", model_metrics)

        # ✅ Add this line to collect user profile + flattened metrics
        paired_profile_metrics.append({
            **profiles[i],
            **model_metrics
        })

        # --- Baseline ---
        baseline_start = time.perf_counter()
        essay = generate_baseline_essay(profiles, merged_question)
        # print(f"Baseline LLM response {i}: {essay}")
        baseline_end = time.perf_counter()
        baseline_latency = baseline_end - baseline_start
        baseline_metrics = evaluate_baseline_response(essay)
        baseline_metrics.update(
            {"type": "baseline", "id": i, "latency_seconds": baseline_latency})
        baseline_records.append(baseline_metrics)
        print(
            f"Baseline metrics (Latency: {baseline_latency:.3f}s):", baseline_metrics)

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


def save_markdown_table(df: pd.DataFrame) -> str:
    markdown = generate_comparison_table_markdown(df)
    with open("./comparison_table.md", "w") as f:
        f.write("# Baseline vs Model Comparison Table\n\n")
        f.write(markdown)
    return "./comparison_table.md"


def save_results_to_csv(df: pd.DataFrame, path: str = None):
    """
    Saves the full evaluation results dataframe to a CSV file with timestamp.

    Args:
        df: DataFrame containing evaluation metrics
        path: Optional custom path where the CSV file will be saved

    Returns:
        Path to the saved CSV file
    """

    # Get current timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory if it doesn't exist
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    # Create filename with timestamp
    if path is None:
        filename = f"eval_results_{timestamp}.csv"
        path = os.path.join(results_dir, filename)

    # Add metadata columns to the dataframe
    df['timestamp'] = timestamp

    # Save to CSV
    df.to_csv(path, index=False)
    print(f"✅ Results saved to: {path}")

    return path


# Identification of correlation between specific attributes, such as age, ethnicity, job , and a specific metric in our setting to analyse qualitatively
# Here is the attribute in each profile
'''
  {
        "age": "95",
        "sex": "Female",
        "race": "White alone",
        "ancestry": "European",
        "household language": "English",
        "education": "Regular high school diploma",
        "employment status": "Not in labor force",
        "class of worker": "Private",
        "industry category": "Education",
        "occupation category": "Teacher",
        "detailed job description": "Retired teacher",
        "income": "0-10000",
        "marital status": "Widowed",
        "household type": "One-person household",
        "family presence and age": "No family",
        "place of birth": "New Jersey/NJ",
        "citizenship": "Born in the United States",
        "veteran status": "Non-Veteran",
        "disability": "With a disability",
        "health insurance": "With health insurance coverage",
        "big five scores": "Openness: Average, Conscientiousness: Low, Extraversion: High, Agreeableness: Extremely High, Neuroticism: High",
        "defining quirks": "Enjoys listening to classical music",
        "mannerisms": "Often seen with reading glasses and a book",
        "personal time": "Spends free time reading or gardening",
        "lifestyle": "Quiet and peaceful",
        "ideology": "Liberal",
        "political views": "Democrat",
        "religion": "Religiously Unaffiliated"
    },
'''
# === LLM-based simplification functions ===


def simplify_attribute(field_name: str, text: str) -> str:
    prompt = (
        f"Given this user's {field_name}, extract a concise descriptor:\\n"
        f"{field_name}: {text}\\n"
        "Single-word label:"
    )
    try:
        return judgeModel.get_completion(prompt).strip().split()[0].lower()
    except:
        return "unknown"


def convert_age_to_bucket(age_str: str) -> str:
    prompt = f"Convert age to bucket (e.g., 'young adult', 'senior'): {age_str}"
    try:
        return judgeModel.get_completion(prompt).strip().split()[0].lower()
    except:
        return "unknown"


def convert_income_to_range(income_str: str) -> str:
    prompt = f"Convert income '{income_str}' to category (e.g., 'low', 'medium', 'high'):"
    try:
        return judgeModel.get_completion(prompt).strip().split()[0].lower()
    except:
        return "unknown"


def simplify_education_level(text: str) -> str:
    prompt = f"Categorize education: {text}"
    try:
        return judgeModel.get_completion(prompt).strip().split()[0].lower()
    except:
        return "unknown"


def infer_boolean_from_text(field: str, text: str) -> bool:
    prompt = (
        f"Given this user's '{field}' field, determine whether it implies a True or False value "
        f"in the context of binary classification (e.g., 'has insurance' vs. 'no insurance').\\n"
        f"Field content: {text}\\n"
        f"Return only True or False:"
    )
    try:
        result = judgeModel.get_completion(prompt).strip().lower()
        return "true" in result
    except:
        return pd.NA


# === Preprocessing user profiles ===
'''
Structured by processing priority:

Special handling for known structured fields (age, income, education, big five scores)

LLM-based simplification for free-text fields (e.g., mannerisms, lifestyle)

For all other string fields, automatically determine via LLM whether:

It is binary → call infer_boolean_from_text

Or categorical → call simplify_attribute

Avoids redundant processing

Adds _bool or _cat suffix to new columns

'''


def preprocess_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    processed_cols = set()

    traits = ["openness", "conscientiousness",
              "extraversion", "agreeableness", "neuroticism"]

    def parse_big_five(text):
        result = {trait: pd.NA for trait in traits}
        if pd.isna(text):
            return result
        parts = [p.strip() for p in text.split(",")]
        for part in parts:
            if ':' in part:
                k, v = part.split(":")
                result[k.strip().lower()] = v.strip().lower()
        return result

    # === 1. Structured processing: Big Five
    if "big five scores" in df.columns:
        parsed = df["big five scores"].apply(parse_big_five)
        for trait in traits:
            df[trait] = parsed.apply(lambda d: d.get(trait, pd.NA))
            processed_cols.add(trait)
        df.drop(columns=["big five scores"], inplace=True)
        processed_cols.add("big five scores")

    # === 2. LLM simplification for descriptive fields
    descriptive_fields = [
        "detailed job description", "lifestyle",
        "defining quirks", "mannerisms", "personal time"
    ]
    for field in descriptive_fields:
        if field in df.columns:
            df[field] = df[field].apply(lambda x: simplify_attribute(
                field, x) if isinstance(x, str) else pd.NA)
            processed_cols.add(field)

    # === 3. Bucketing age, income, education
    if "age" in df.columns:
        df["age"] = df["age"].apply(
            lambda x: convert_age_to_bucket(str(x)) if pd.notna(x) else pd.NA)
        processed_cols.add("age")

    if "income" in df.columns:
        df["income"] = df["income"].apply(
            lambda x: convert_income_to_range(str(x)) if pd.notna(x) else pd.NA)
        processed_cols.add("income")

    if "education" in df.columns:
        df["education"] = df["education"].apply(
            lambda x: simplify_education_level(str(x)) if pd.notna(x) else pd.NA)
        processed_cols.add("education")

    # === 4. For other string fields, auto-infer type using LLM
    for col in df.columns:
        if col in processed_cols:
            continue
        if df[col].dtype != "object" or df[col].isna().all():
            continue

        try:
            sample_val = df[col].dropna().iloc[0]
            if not isinstance(sample_val, str):
                continue

            # Ask LLM whether this is a binary or categorical field
            prompt = (
                f"Based on the user's '{col}' field, determine whether this is a binary (True/False) field "
                f"or a multi-class categorical attribute.\n"
                f"Example value: {sample_val}\n"
                f"Return only one word: 'binary' or 'categorical'."
            )
            decision = judgeModel.get_completion(prompt).strip().lower()

            if decision == "binary":
                df[col + "_bool"] = df[col].apply(lambda x: infer_boolean_from_text(
                    col, x) if isinstance(x, str) else pd.NA)
            elif decision == "categorical":
                df[col + "_cat"] = df[col].apply(lambda x: simplify_attribute(
                    col, x) if isinstance(x, str) else pd.NA)

            processed_cols.add(col)

        except Exception as e:
            print(f"⚠️ Skipping column `{col}` due to error: {e}")

    return df


# === Correlation analysis with markdown summary ===

def analyze_attribute_correlations(
    input: list,
    output_dir="./correlation_analysis",
    corr_threshold=0.3,
    save_global_summary=True
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not input:
        print("⚠️ Empty input list.")
        return

    df = pd.DataFrame(input)
    df = df[df.get("type", "") == "model"].copy()

    if "user_profile" in df.columns:
        df = df.drop(columns=["user_profile"])

    metric_cols = [
        "avg_response_length", "response_completeness", "cultural_alignment_score",
        "diversity_entropy", "sensitivity_coverage", "sensitive_topic_mention_rate"
    ]
    metric_cols = [col for col in metric_cols if col in df.columns]

    # Use enhanced preprocessing
    df = preprocess_user_profiles(df)

    # Dynamically determine non-metric profile fields
    profile_cols = [
        col for col in df.columns if col not in metric_cols and col not in {"type", "id"}]

    # Group profile attributes by heuristics for visualization
    grouped_attrs = {
        "Demographic & Identity": [col for col in profile_cols if any(k in col for k in ["age", "sex", "race", "ancestry", "birth", "citizenship", "religion"])],
        "Socioeconomic & Occupational": [col for col in profile_cols if any(k in col for k in ["education", "employment", "income", "occupation", "industry", "class of worker"])],
        "Health & Lifestyle": [col for col in profile_cols if any(k in col for k in ["health", "disability", "veteran", "lifestyle", "personal time"])],
        "Family & Household Context": [col for col in profile_cols if any(k in col for k in ["marital", "household", "family"])],
        "Psychological & Ideological": [col for col in profile_cols if any(k in col for k in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "ideology", "political", "quirks", "mannerisms"])],
    }

    global_top_corrs = []

    for group_name, group_cols in grouped_attrs.items():
        group_cols_present = [col for col in group_cols if col in df.columns]
        if not group_cols_present:
            continue

        df_encoded = df[group_cols_present + metric_cols].copy()
        for col in group_cols_present:
            if df_encoded[col].dtype == "object":
                df_encoded[col] = df_encoded[col].astype("category").cat.codes

        corr_matrix = df_encoded.corr().loc[group_cols_present, metric_cols]
        corr_matrix.to_csv(os.path.join(
            output_dir, f"{group_name.lower().replace(' ', '_')}_correlation.csv"))

        corr_melted = corr_matrix.reset_index().melt(
            id_vars="index", var_name="metric", value_name="correlation")
        corr_melted.rename(columns={"index": "attribute"}, inplace=True)
        corr_melted["abs_corr"] = corr_melted["correlation"].abs()

        top_corr = (
            corr_melted[corr_melted["abs_corr"] >= corr_threshold]
            .sort_values(["metric", "abs_corr"], ascending=[True, False])
            .groupby("metric")
            .head(3)
        )

        if save_global_summary:
            top_corr["group"] = group_name
            global_top_corrs.append(top_corr)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_corr, x="metric",
                    y="correlation", hue="attribute")
        plt.title(
            f"Top Correlations (|r| ≥ {corr_threshold}) per Metric: {group_name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f"{group_name.lower().replace(' ', '_')}_top_corr_plot.png"))
        plt.close()

    if save_global_summary and global_top_corrs:
        all_top = pd.concat(global_top_corrs, ignore_index=True)
        csv_path = os.path.join(output_dir, "global_top_correlations.csv")
        all_top.to_csv(csv_path, index=False)
        generate_enhanced_markdown_summary(csv_path, os.path.join(
            output_dir, "global_top_correlations.md"), output_dir)


# === Markdown generator ===
def generate_enhanced_markdown_summary(summary_csv, output_md, base_dir):
    if not os.path.exists(summary_csv):
        print(f"❌ Summary CSV not found: {summary_csv}")
        return

    df = pd.read_csv(summary_csv)
    if "abs_corr" not in df.columns:
        df["abs_corr"] = df["correlation"].abs()

    metric_commentary = {
        "cultural_alignment_score": "Measures how well the response aligns with the user's cultural and ideological framing.",
        "sensitivity_coverage": "Assesses how thoroughly the response engages with sensitive or identity-relevant content.",
        "sensitive_topic_mention_rate": "Rate at which sensitive topics are explicitly mentioned.",
        "diversity_entropy": "Lexical or topical entropy — higher values imply broader or more varied responses.",
        "response_completeness": "Degree to which the response fully addresses the query.",
        "avg_response_length": "Token-level response length — often reflects verbosity or depth.",
    }

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# 🧠 Correlation Summary Report\\n\\n")
        f.write("This report summarizes the strongest observed correlations between **simplified user profile attributes** and model evaluation metrics.\\n\\n")
        f.write(
            "Each section links to the full correlation data and visual summary.\\n\\n")

        for metric in df["metric"].unique():
            df_metric = df[df["metric"] == metric].copy()
            df_metric["abs_corr"] = df_metric["correlation"].abs()
            df_sorted = df_metric.sort_values(
                by="correlation", ascending=False)

            f.write(f"## 📊 Metric: `{metric}`\\n")
            f.write(
                f"**Interpretation**: {metric_commentary.get(metric, 'No interpretation available.')}\\n\\n")

            group_name = df_metric["group"].iloc[0]
            group_key = group_name.lower().replace(" ", "_")
            f.write(
                f"📎 [Full correlation CSV](./{group_key}_correlation.csv)\\n")
            f.write(
                f"🖼️ [Top 3 correlation plot](./{group_key}_top_corr_plot.png)\\n\\n")

            top_pos = df_sorted.iloc[0]
            f.write("**🔼 Top Positive Correlation**\\n")
            f.write(
                f"- **Attribute**: `{top_pos['attribute']}` from *{top_pos['group']}*\\n")
            f.write(
                f"- **Correlation**: r = {top_pos['correlation']:.2f}\\n\\n")

            top_neg = df_metric.sort_values(
                by="correlation", ascending=True).iloc[0]
            f.write("**🔽 Top Negative Correlation**\\n")
            f.write(
                f"- **Attribute**: `{top_neg['attribute']}` from *{top_neg['group']}*\\n")
            f.write(
                f"- **Correlation**: r = {top_neg['correlation']:.2f}\\n\\n")

            f.write("**📌 Notable Correlations (Top 3 by absolute value):**\\n")
            top3 = df_metric.sort_values(
                by="abs_corr", ascending=False).head(3)
            for _, row in top3.iterrows():
                trend = "increase" if row["correlation"] > 0 else "decrease"
                f.write(
                    f"- `{row['attribute']}` (`{row['group']}`): r = {row['correlation']:.2f} → likely {trend} in `{metric}`\\n")

            f.write("\\n---\\n\\n")

    print(
        f"✅ Enhanced correlation summary Markdown report saved to: {output_md}")


def save_paired_profiles_to_json(data, filename="paired_profiles_metrics.json"):
    """
    Saves the paired profile metrics to a JSON file.

    Args:
        data: List of dictionaries containing paired profile and metrics data
        filename: Name of the output JSON file

    Returns:
        Path to the saved JSON file
    """
    # Get current timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory if it doesn't exist
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    # Create filename with timestamp
    filename_with_timestamp = f"{filename.split('.')[0]}_{timestamp}.json"
    path = os.path.join(results_dir, filename_with_timestamp)

    # Convert data to JSON and save
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Paired profiles with metrics saved to: {path}")
    return path


if __name__ == "__main__":
    df_results = compare_with_baseline(n=10)  # Adjust n as needed
    path = save_markdown_table(df_results)
    save_results_to_csv(df_results)
    print(f"\n✅ Markdown saved to: {path}")

    # Save paired profiles to JSON for visualization
    profiles_path = save_paired_profiles_to_json(paired_profile_metrics)

    analyze_attribute_correlations(paired_profile_metrics, corr_threshold=0.3)
