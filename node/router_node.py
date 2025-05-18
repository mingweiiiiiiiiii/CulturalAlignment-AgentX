import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Any

# Mocked classes (replace with your real imports)
from node.cultural_expert_node import CulturalExpertManager
from node.sen_agent_node import determine_cultural_sensitivity
from node.embed_utils import embed_persona
from utility.measure_time import measure_time
import requests
# === Embedding Function ===
"""def embed_persona(persona: Dict[str, Any]) -> np.ndarray:
    text = ", ".join(f"{k}: {v}" for k, v in persona.items())
    response = requests.post("http://localhost:8000/embeddings/", json={"text": text})
    if response.status_code != 200:
        print(f"⚠️ Failed to get embedding: {response.text}")
        return np.zeros(768)
    embedding = response.json().get("embedding", [])
    return np.array(embedding)"""

# === Router Function ===
@measure_time
def route_to_cultures(
    state: Dict[str, Any],
    lambda_1: float = 0.6,
    lambda_2: float = 0.4,
    top_k: int = 3,
    tau: float = -30.0,
    precomputed_centroids: np.ndarray = None,
) -> List[Dict[str, Any]]:

    manager = CulturalExpertManager(state=state)

    # Prepare input
    q = state["question_meta"]["original"]
    user_profile = state.get("user_profile", {})
    user_embedding = embed_persona(user_profile)
    sensitive_topics = state["question_meta"].get("sensitive_topics", [])

    # Step 1: Embed experts
    # Generate experts_list and expert_embeddings
    expert_list, expert_embeddings = manager.get_all_persona_embeddings()

    # Step 2: Topic embedding
    if sensitive_topics:
        topic_embeddings = [
            embed_persona({"country": topic}) for topic in sensitive_topics
        ]
    else:
        topic_embeddings = [user_embedding]

    T = np.stack(topic_embeddings)
    t_bar = np.mean(T, axis=0)
    z = (lambda_1 * t_bar + lambda_2 * user_embedding) / (lambda_1 + lambda_2)

    # Step 3: Score experts
    scores = -np.sum(np.abs(expert_embeddings - z), axis=1)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    s_top = scores[top_indices]

    # Step 4: Fallback if needed
    if np.max(s_top) < tau:
        if precomputed_centroids is None:
            kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
            kmeans.fit(expert_embeddings)
            centroids = kmeans.cluster_centers_
        else:
            centroids = precomputed_centroids

        closest_centroid_idx = np.argmin(
            np.linalg.norm(user_embedding - centroids, axis=1)
        )
        closest_centroid = centroids[closest_centroid_idx]

        scores = -np.sum(np.abs(expert_embeddings - closest_centroid), axis=1)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        s_top = scores[top_indices]

    # Step 5: Softmax normalize
    s_max = np.max(s_top)
    logits = np.clip(s_top - s_max, -100, 100)
    softmax_weights = np.exp(logits) / np.sum(np.exp(logits))

    # Step 6: Generate selected expert set
    selected_experts = [
      {"culture": expert_list[i], "weight": float(softmax_weights[j])}
      for j, i in enumerate(top_indices)
    ]

    state["question_meta"]["relevant_cultures"] = [e["culture"] for e in selected_experts]
    state["activate_router"] = False
    return {
    "activate_set": selected_experts
    }


# === Test Case ===
if __name__ == "__main__":
    input_state = {
        "question_meta": {
            "original": "How should conflicts be resolved in a community?",
            "sensitive_topics": [],
            "relevant_cultures": [],
        },
        "user_profile": {
            "id": "user123",
            "demographics": {
                "sex": "Female",
                "age": 28,
                "marital_status": "Single",
                "education": "Master's Degree",
                "employment_sector": "Healthcare",
                "social_class": "Middle",
                "income_level": "Medium",
                "ethnicity": "Asian",
                "country": "India",
            },
            "preferences": {},
        },
    }
    output = route_to_cultures(state=input_state, top_k=2)

    print("\n=== Selected Experts ===")
    for expert in output:
        print(
            f"\nCulture: {expert['culture']}\nWeight: {expert['weight']:.4f}\nPrompt:\n{expert['prompt']}"
        )
