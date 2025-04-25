import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from typing import Dict, List
from node.types import GraphState
from sklearn.metrics.pairwise import cosine_similarity

def route_to_cultures(
    state: GraphState, 
    evaluator: EvaluationLosses,
    expert_list: List[str],
    expert_embeddings: np.ndarray,
    prompt_libraries: Dict[str, List[str]],
    lambda_1: float = 0.6,
    lambda_2: float = 0.4,
    top_k: int = 3,
    tau: float = -30.0
) -> Dict:
) -> Dict:
    q = state["question_meta"]["original"]
    user_profile = state["user_profile"]
    user_embedding = state["user_embedding"]
    sensitive_topics = state["question_meta"].get("sensitive_topics", [])

    # Embed topic countries using WorldValueSurveyProcess
    from WorldValueSurveyProcess import embed_profile
    topic_embeddings = [embed_profile({
        "sex": "N/A", "age": 30, "marital_status": "N/A",
        "education": "N/A", "employment_sector": "N/A",
        "social_class": "N/A", "income_level": "N/A",
        "ethnicity": "N/A", "country": topic
    }) for topic in sensitive_topics] if sensitive_topics else [user_embedding]

    T = np.stack(topic_embeddings)
    t_bar = np.mean(T, axis=0)
    z = (lambda_1 * t_bar + lambda_2 * user_embedding) / (lambda_1 + lambda_2)

    # Calculate similarity scores (Manhattan distance)
    scores = -np.sum(np.abs(expert_embeddings - z), axis=1)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    s_top = scores[top_indices]

    # Fallback to KMeans if top score is too low
    if np.max(s_top) < tau:
        kmeans = KMeans(n_clusters=3, random_state=0).fit(expert_embeddings)
        centroids = kmeans.cluster_centers_
        closest_cluster = np.argmin(np.linalg.norm(centroids - user_embedding, axis=1))
        z = centroids[closest_cluster]
        scores = -np.sum(np.abs(expert_embeddings - z), axis=1)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        s_top = scores[top_indices]

    s_max = np.max(s_top)
    softmax_weights = np.exp(s_top - s_max) / np.sum(np.exp(s_top - s_max))

    # Prepare weighted prompts for selected experts
    A = []
    for i, idx in enumerate(top_indices):
        culture = expert_list[idx]
        weight = softmax_weights[i]
        prompt = generate_expert_prompt(user_profile, q, sensitive_topics[0] if sensitive_topics else "general", culture)
        A.append((culture, weight, prompt))

    relevant_cultures = [culture for culture, _, _ in A]
    state.update({
        "question_meta": {
            **state["question_meta"],
            "relevant_cultures": relevant_cultures
        },
        "current_state": "router",
        "expert_weights_and_prompts": A
    })
    return state


def cultural_expert_node_factory(culture_name: str):
    def expert_fn(state: GraphState) -> Dict:
        question = state["question_meta"]["original"]
        response = model(f"As a representative of {culture_name} culture, how would you answer: '{question}'?")
        updated = state.get("response_state", {}).get("expert_responses", [])
        new_entry = {"culture": culture_name, "response": response}
        return {
            "response_state": {"expert_responses": updated + [new_entry]},
            "current_state": f"expert_{culture_name}"
        }
    return expert_fn
