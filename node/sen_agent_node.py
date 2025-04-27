import numpy as np
import re
from typing import Dict
import ollama
from sklearn.metrics.pairwise import cosine_similarity

# === Reference bank of example questions with sensitivity levels ===
reference_bank = {
    "Religion is outdated and should be banned.": "high",
    "Why do women complain so much?": "high",
    "Why do Chinese people eat weird food?": "medium",
    "Why do Muslims pray so often?": "medium",
    "What is the capital of Germany?": "low",
    "Tell me about Christmas traditions.": "low",
    "Why are African countries poor?": "high",
    "Are Asian people better at math?": "medium",
    "Describe the culture of Brazil.": "low",
    "Is it strange that men cry?": "medium",
    "Is English superior to other languages?": "high",
    "Talk about Indian festivals.": "low",
    "Why do people in the Middle East fight so much?": "high",
    "Why do immigrants refuse to assimilate?": "high",
    "Explain traditional Japanese tea ceremonies.": "low",
    "Why are Europeans so formal?": "medium",
    "Why are indigenous traditions outdated?": "high",
    "Are Latinos always passionate?": "medium",
    "Tell me about French cuisine.": "low",
    "Why do some African tribes still exist today?": "high"
}

sensitivity_scale = {"low": 3, "medium": 6, "high": 9}

# === Precompute reference embeddings (flattened) ===
reference_embeddings = {}
for q in reference_bank:
    emb = ollama.embed(model="mxbai-embed-large", input=q)["embeddings"]
    reference_embeddings[q] = np.array(emb).flatten()

reference_matrix = np.stack(list(reference_embeddings.values()))

# === Determine cultural sensitivity ===
def determine_cultural_sensitivity(state) -> Dict:
    question = state["question_meta"]["original"]
    question_embedding = np.array(ollama.embed(model="mxbai-embed-large", input=question)["embeddings"]).flatten()

    # Calculate cosine similarity between the input and all references
    sims = cosine_similarity(question_embedding.reshape(1, -1), reference_matrix)[0]

    # Map reference questions to similarity scores
    sims_dict = dict(zip(reference_bank.keys(), sims))

    # Find best match
    best_match, best_sim = max(sims_dict.items(), key=lambda x: x[1])
    base_score = sensitivity_scale.get(reference_bank[best_match], 0)

    # Adjust final score
    sensitivity_score = min(10, max(0, int(base_score + 2 * (best_sim - 0.5) * 3)))

    return {
        "question_meta": {
            **state["question_meta"],
            "is_sensitive": sensitivity_score >= 5,
            "sensitivity_score": sensitivity_score,
            "nearest_sensitive_example": best_match
        },
        "current_state": "sensitivity_check"
    }

# === Test ===
if __name__ == "__main__":
    mock_state = {
        "question_meta": {
            "original": "Why do women complain so much?"
        }
    }
    print("Starting sensitivity check...")
    try:
        result = determine_cultural_sensitivity(mock_state)
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")
