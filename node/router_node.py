import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Any
import ollama

# Mocked classes (replace with your real imports)
from node.cultural_expert_node import CulturalExpertManager
from node.sen_agent_node import determine_cultural_sensitivity
from utility.measure_time import measure_time

# === Embedding Function ===
def embed_persona(persona: Dict[str, Any]) -> np.ndarray:
    text = ", ".join(f"{k}: {v}" for k, v in persona.items())
    response = ollama.embed(model="mxbai-embed-large", input=text)
    embeddings = response.get("embeddings", [])

    if not embeddings:
        print(f"🔍 Text passed to embedding model:\n{text}")
        print(f"⚠️ No embeddings returned for text: {text}")
        return np.zeros(768)  # fallback: return zero vector of expected size

    return np.array(embeddings[0])


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

    # Setup experts

    manager = CulturalExpertManager(state=state)

    # Generate experts
    expert_instances = manager.generate_expert_instances()
    expert_list = list(expert_instances.keys())

    # Sensitivity detection
    sensitivity_info = determine_cultural_sensitivity(state)
    state["question_meta"].update(sensitivity_info["question_meta"])

    # Prepare input
    q = state["question_meta"]["original"]
    user_profile = state.get("user_profile", {})
    user_embedding = embed_persona(user_profile)
    sensitive_topics = state["question_meta"].get("sensitive_topics", [])

    # Step 1: Embed experts
    dict_expert_embeddings = {}
    dict_expert_prompt_text = {}

    for expert_name in expert_list:
        expert = expert_instances[expert_name]
        generated_response = expert.generate_response(q)
        dict_expert_embeddings[expert_name] = embed_persona(
            {"response": generated_response}
        )
        dict_expert_prompt_text[expert_name] = generated_response

    expert_embeddings = np.stack(list(dict_expert_embeddings.values()))

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
    selected_experts = []
    for i, idx in enumerate(top_indices):
        culture = expert_list[idx]
        weight = softmax_weights[i]
        prompt_text = dict_expert_prompt_text[culture]  # ✅ Correct fetching prompt
        selected_experts.append(
            {"culture": culture, "weight": float(weight), "prompt": prompt_text}
        )
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
