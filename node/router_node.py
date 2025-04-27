import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Any
import random
import ollama

# 🆕 Import the sensitivity check function
from sen_agent_node import determine_cultural_sensitivity

# ===============================
# 🚀 Embedding Function
# ===============================
def embed_persona(persona: Dict[str, Any]) -> np.ndarray:
    """Embed a persona dictionary into a vector using Ollama embedding."""
    if not isinstance(persona, dict):
        raise TypeError("Persona must be a dictionary.")

    for k, v in persona.items():
        if not isinstance(k, str):
            raise TypeError("All keys must be strings.")
        if not isinstance(v, (str, int, float)):
            raise TypeError("All values must be string, int, or float.")

    text = ", ".join(f"{k}: {v}" for k, v in persona.items())
    response = ollama.embed(model="mxbai-embed-large", input=text)

    if "embeddings" not in response:
        raise KeyError(f"'embeddings' not found in response: {response}")
    
    embedding = np.array(response["embeddings"][0])

    if embedding.ndim != 1:
        raise ValueError(f"Expected 1D embedding vector, got shape {embedding.shape}")

    return embedding

# ===============================
# 🛠️ Routing Function
# ===============================
class GraphState(dict):
    """A simple extension of dict for holding graph states."""
    pass

def route_to_cultures(
    state, 
    expert_list: List[str],
    expert_embeddings: np.ndarray,
    prompt_libraries: Dict[str, List[str]],
    lambda_1: float = 0.6,
    lambda_2: float = 0.4,
    top_k: int = 3,
    tau: float = -30.0,
    precomputed_centroids: np.ndarray = None
) -> Dict:
    """Top-k Cultural Expert Routing with fallback clustering."""

    # 🛠️ Correctly update the original state with sensitivity info
    sensitivity_info = determine_cultural_sensitivity(state)
    state.update(sensitivity_info)

    q = state["question_meta"]["original"]
    user_profile = state["user_profile"]
    user_embedding = state["user_embedding"]
    sensitive_topics = state["question_meta"].get("sensitive_topics", [])

    # 🆙 Adjust parameters based on sensitivity score
    sensitivity_score = state["question_meta"].get("sensitivity_score", 0)
    if sensitivity_score >= 8:
        tau = -10.0
        top_k = min(top_k, 2)
    elif sensitivity_score >= 5:
        tau = -20.0
    else:
        tau = -30.0

    inv_lambda = 1.0 / (lambda_1 + lambda_2)

    # Embed topics
    if sensitive_topics:
        topic_embeddings = [embed_persona({
            "sex": "N/A", "age": 30, "marital_status": "N/A",
            "education": "N/A", "employment_sector": "N/A",
            "social_class": "N/A", "income_level": "N/A",
            "ethnicity": "N/A", "country": topic
        }) for topic in sensitive_topics]
    else:
        topic_embeddings = [user_embedding]

    T = np.stack(topic_embeddings)
    t_bar = np.mean(T, axis=0)
    z = inv_lambda * (lambda_1 * t_bar + lambda_2 * user_embedding)

    # Distance scoring (Manhattan distance)
    scores = -np.sum(np.abs(expert_embeddings - z), axis=1)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    s_top = scores[top_indices]

    # Fallback to cluster centroids if needed
    if np.max(s_top) < tau:
        if precomputed_centroids is None:
            kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
            kmeans.fit(expert_embeddings)
            centroids = kmeans.cluster_centers_
        else:
            centroids = precomputed_centroids

        closest_cluster_idx = np.argmin(np.sum(np.abs(centroids - user_embedding), axis=1))
        z = centroids[closest_cluster_idx]

        scores = -np.sum(np.abs(expert_embeddings - z), axis=1)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        s_top = scores[top_indices]

    # Numerically stable softmax
    s_max = np.max(s_top)
    logits = np.clip(s_top - s_max, -100, 100)
    softmax_weights = np.exp(logits) / np.sum(np.exp(logits))

    # Assemble prompts
    A = []
    for i, idx in enumerate(top_indices):
        culture = expert_list[idx]
        weight = softmax_weights[i]
        topic_for_prompt = sensitive_topics[0] if sensitive_topics else "general"

        # Select real prompt from library
        base_prompt = random.choice(prompt_libraries[culture])

        # Combine with user info without formatting the real prompt
        prompt = f"User profile: {user_profile}\nQuery: {q}\nTopic: {topic_for_prompt}\n\n{base_prompt}"

        A.append((culture, weight, prompt))

    relevant_cultures = [culture for culture, _, _ in A]

    # Update state
    state.update({
        "question_meta": {
            **state["question_meta"],
            "relevant_cultures": relevant_cultures
        },
        "current_state": "router",
        "expert_weights_and_prompts": A
    })

    return state

# ===============================
# 🧪 Test Case
# ===============================
if __name__ == "__main__":
    user_profile = {
        "sex": "female",
        "age": 27,
        "marital_status": "single",
        "education": "PhD",
        "employment_sector": "Healthcare",
        "social_class": "upper",
        "income_level": "high",
        "ethnicity": "Latina",
        "country": "Spain"
    }

    expert_list = [f"Culture_{i}" for i in range(5)]
    expert_embeddings = np.random.rand(5, 1024)
    prompt_libraries = {
        culture: [
            f"Discuss leadership values from the {culture} perspective.",
            f"Describe how cultural background shapes leadership in {culture}."
        ] for culture in expert_list
    }
    user_embedding = np.random.rand(1024)

    state = GraphState({
        "question_meta": {
            "original": "Why do women complain so much?",  # <-- sensitive question
            "sensitive_topics": ["Spain", "Mexico"]
        },
        "user_profile": user_profile,
        "user_embedding": user_embedding
    })

    updated_state = route_to_cultures(
        state,
        expert_list,
        expert_embeddings,
        prompt_libraries
    )

    print("\n✅ Updated State after Routing:")
    for key, value in updated_state.items():
        print(f"{key}: {value}")
