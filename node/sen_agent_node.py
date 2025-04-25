import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from typing import Dict, List
from node.types import GraphState
from sklearn.metrics.pairwise import cosine_similarity

# === Agent Nodes ===

# === Reference bank of example questions with sensitivity levels ===
# RAG 
reference_bank = {
    "Religion is outdated and should be banned.": "high",
    "Why do women complain so much?": "high",
    "Why do Chinese people eat weird food?": "medium",
    "Why do Muslims pray so often?": "medium",
    "What is the capital of Germany?": "low",
    "Tell me about Christmas traditions.": "low"
}

sensitivity_scale = {"low": 3, "medium": 6, "high": 9}
reference_embeddings = {
    q: get_text_embedding(q) for q in reference_bank
}
# use the theory of this paper into this function
# https://arxiv.org/abs/1807.03748
def determine_cultural_sensitivity(state: GraphState, evaluator: EvaluationLosses) -> Dict:
    question = state["question_meta"]["original"]
    question_embedding = get_text_embedding(question)

    # Calculate cosine similarity to each reference
    sims = {
        ref_q: cosine_similarity(
            [question_embedding], [ref_emb]
        )[0][0]
        for ref_q, ref_emb in reference_embeddings.items()
    }

    # Find top-matched example and base score
    best_match, best_sim = max(sims.items(), key=lambda x: x[1])
    base_score = sensitivity_scale[reference_bank[best_match]]

    # Adjust final score: weight similarity and base sensitivity
    # Similarity [0, 1] → scaled adjustment [-1, +1]
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
