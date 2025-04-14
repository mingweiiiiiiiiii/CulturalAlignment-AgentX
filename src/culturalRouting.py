import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from typing import List, Dict
import unittest

# === Load BERT model and tokenizer once ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# === BERT-based text embedding ===
@torch.no_grad()
def get_text_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().numpy()  # Shape: (768,)

# === Prompt generation ===
def gen_prompt(question: str, topic_mean: np.ndarray, prompt_library: List[str]) -> str:
    return f"{prompt_library[0]}\n\n{question}"

# === Mock expert classes for testing ===
expert_classes = {
    "US": None,
    "China": None,
    "India": None
}

# === Main routing function based on LaTeX algorithm ===
def route_to_cultures(state: Dict, lambda_1: float = 0.6, lambda_2: float = 0.4) -> Dict:
    q = state["question_meta"]["original"]
    d_u = get_text_embedding(str(state.get("user_profile", "")))
    sensitive_topics = state["question_meta"].get("sensitive_topics", [])

    # Construct topic matrix D_T
    topic_vectors = [get_text_embedding(topic) for topic in sensitive_topics]
    if not topic_vectors:
        topic_vectors = [get_text_embedding("general")]
    D_T = np.stack(topic_vectors)

    # Step 1: Mean topic vector \bar{d}_T
    d_T_bar = np.mean(D_T, axis=0)

    # Step 2: Topic-user centroid d_TU
    d_TU = (lambda_1 * d_T_bar + lambda_2 * d_u) / (lambda_1 + lambda_2)

    # Step 3: Expert matrix D_E
    culture_list = list(expert_classes.keys())
    D_E = np.stack([get_text_embedding(culture) for culture in culture_list])

    # Step 4: Similarity vector S (negative L1 distance)
    S = -np.sum(np.abs(D_E - d_TU), axis=1)

    # Step 5: Top-k indices \mathcal{I}_k
    k = min(3, len(culture_list))
    topk_indices = np.argsort(S)[-k:][::-1]

    # Step 6: Top-k similarity scores S_k
    S_k = S[topk_indices]

    # Step 7: Numerical stability trick s_max
    s_max = np.max(S_k)

    # Step 8: Softmax weights w
    exp_scores = np.exp(S_k - s_max)
    softmax_weights = exp_scores / np.sum(exp_scores)

    # Step 9: Initialize active expert set A
    A = []

    # Step 10: For each top-k expert j
    for idx, weight in zip(topk_indices, softmax_weights):
        culture = culture_list[idx]
        P_j = [f"As a representative of {culture}, please consider:"]
        prompt_j = gen_prompt(q, d_T_bar, P_j)
        A.append((culture, weight, prompt_j))

    # Final output
    relevant_cultures = [culture for culture, _, _ in A]
    return {
        "question_meta": {
            **state["question_meta"],
            "relevant_cultures": relevant_cultures
        },
        "current_state": "router",
        "expert_weights_and_prompts": A
    }

# === Example usage ===
if __name__ == "__main__":
    example_state = {
        "question_meta": {
            "original": "How should we address gender roles in different societies?",
            "sensitive_topics": ["gender", "roles"]
        },
        "user_profile": {
            "id": "user_123",
            "demographics": {"age": 30, "region": "Europe"},
            "preferences": {"tone": "academic"}
        }
    }

    result = route_to_cultures(example_state, lambda_1=0.5, lambda_2=0.5)
    print("Selected Cultures:", result["question_meta"]["relevant_cultures"])
    print("Expert Weights & Prompts:")
    for culture, weight, prompt in result["expert_weights_and_prompts"]:
        print(f"{culture}: weight={weight:.4f}, prompt=\"{prompt}\"")

# === Unit Tests ===
class TestRoutingFunction(unittest.TestCase):
    def setUp(self):
        self.state = {
            "question_meta": {
                "original": "What are cultural perceptions of family hierarchy?",
                "sensitive_topics": ["family", "hierarchy"]
            },
            "user_profile": {
                "id": "user_456",
                "demographics": {"region": "Asia"},
                "preferences": {"tone": "neutral"}
            }
        }

    def test_output_structure(self):
        result = route_to_cultures(self.state, lambda_1=0.5, lambda_2=0.5)
        self.assertIn("question_meta", result)
        self.assertIn("relevant_cultures", result["question_meta"])
        self.assertIn("expert_weights_and_prompts", result)

    def test_number_of_experts(self):
        result = route_to_cultures(self.state)
        self.assertLessEqual(len(result["expert_weights_and_prompts"]), 3)

    def test_softmax_sum(self):
        result = route_to_cultures(self.state)
        weights = [w for _, w, _ in result["expert_weights_and_prompts"]]
        self.assertAlmostEqual(sum(weights), 1.0, places=4)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)