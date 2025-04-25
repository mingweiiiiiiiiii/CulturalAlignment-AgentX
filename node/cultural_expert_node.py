import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from typing import Dict, List
from node.types import GraphState
from sklearn.metrics.pairwise import cosine_similarity

# === Load BERT model and tokenizer once ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()


# === Mock expert classes for testing ===
expert_classes = {
    "US": None,
    "China": None,
    "India": None
}


# === BERT-based text embedding ===
@torch.no_grad()
def get_text_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().numpy()  # Shape: (768,)


# === Cultural Expert Abstract Base Class ===
class CulturalExpert(ABC):
    def __init__(self, culture_name: str):
        self.culture_name = culture_name

    @abstractmethod
    def generate_response(self, question: str) -> str:
        pass

    def __call__(self, state: GraphState) -> Dict:
        question = state["question_meta"]["original"]
        response = self.generate_response(question)
        updated = state.get("response_state", {}).get("expert_responses", [])
        new_entry = {"culture": self.culture_name, "response": response}
        return {
            "response_state": {"expert_responses": updated + [new_entry]},
            "current_state": f"expert_{self.culture_name}"
        }

# === Cultural Expert Implementations ===
class USExpert(CulturalExpert):
    def __init__(self):
        super().__init__("US")

    def generate_response(self, question: str) -> str:
        return model(f"As a representative of US culture, how would you answer: '{question}'?")

class ChineseExpert(CulturalExpert):
    def __init__(self):
        super().__init__("China")

    def generate_response(self, question: str) -> str:
        return model(f"As a representative of Chinese culture, how would you answer: '{question}'?")

class IndianExpert(CulturalExpert):
    def __init__(self):
        super().__init__("India")

    def generate_response(self, question: str) -> str:
        return model(f"As a representative of Indian culture, how would you answer: '{question}'?")
