import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from typing import Dict, List
from .types import GraphState
from sklearn.metrics.pairwise import cosine_similarity
# === Global counter for planner ===
planner_counter = 0

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




# === Prompt generation ===
def gen_prompt(question: str, topic_mean: np.ndarray, prompt_library: List[str]) -> str:
    return f"{prompt_library[0]}\n\n{question}"

# === Mock expert classes for testing ===
expert_classes = {
    "US": None,
    "China": None,
    "India": None
}

# === Mock LLM model - replace with actual implementation ===
def model(prompt: str) -> str:
    print(f"Model prompt: {prompt}")
    if "sensitivity score" in prompt:
        return "7"
    elif "components" in prompt:
        return "religion, gender"
    elif "which 2-3 cultures" in prompt:
        return "US, China"
    elif "representative" in prompt:
        return f"This is a {prompt.split()[4]} perspective."
    else:
        return "Aggregated culturally respectful answer."

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
def determine_cultural_sensitivity(state: GraphState) -> Dict:
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


def extract_sensitive_topics(state: GraphState) -> Dict:
    prompt = f"""Identify potentially insensitive language or generalizations in the input.
    Focus on the core subject matter that could potentially involve:
    - stereotypes 
    - biases
    - disrespectful language
    - misunderstandings related to cultural practices, traditions, or specific groups of people.

    Output can contains one to many topics, separated by /

    Examples:
        Input: "I saw a group of people wearing those funny-looking hats at the temple today. What's the deal with that?"
        Topic: Religious Attire/Cultural Practices
        Input: "Why do Chinese always eat Stinky Tofu? It smells so strong."
        Topic: Stereotyping/Food Prejudice
        Input: "Is it true that in Japan, women aren't allowed to work after married?"
        Topic: Gender Roles/Cultural Misinformation/Generalization
        Input: "I don't understand why Asians make such a big deal about Lunar New Year. It seems so unnecessary."
        Topic: Cultural Value Systems/Dismissal of Tradition
        Input: "They all look the same to me. How do you even tell them apart?"
        Topic: Racial Bias/Microaggression

    Input:
    ```
    {question}
    ```     
    """
    question = state["question_meta"]["original"]
    response = model(f"What are the culturally sensitive components in the following question: '{question}'? List them.")
    topics = re.findall(r"[A-Za-z]+", response)
    return {
        "question_meta": {**state["question_meta"], "sensitive_topics": topics},
        "db_action": "write",
        "db_key": "sensitive_topics",
        "db_value": topics,
        "__next__": "database",
        "current_state": "extract_topics"
    }


def planner_agent(state: GraphState) -> Dict:


    global planner_counter
    planner_counter += 1
    counter = planner_counter
    updated_state = {"current_state": "planner", "planner_counter": counter}

    if counter == 1:
        return {
            **updated_state,
            "activate_sensitivity_check": True,
            "__next__": "sensitivity_check"
        }
    elif counter == 2 and state.get("activate_sensitivity_check"):
        return {
            **updated_state,
            "activate_sensitivity_check": False,
            "activate_extract_topics": True,
            "__next__": "extract_topics"
        }
    elif counter == 3 and state.get("activate_extract_topics"):
        return {
            **updated_state,
            "activate_extract_topics": False,
            "db_action": "read",
            "db_key": "sensitive_topics",
            "__next__": "database"
        }
    elif counter == 4 and state.get("db_result") is not None:
        return {
            **updated_state,
            "question_meta": {**state["question_meta"], "sensitive_topics": state["db_result"]},
            "activate_router": True,
            "__next__": "router"
        }
    elif counter == 5 and state.get("response_state", {}).get("expert_responses"):
        responses = state["response_state"]["expert_responses"]
        summary = "\n".join([f"{r['culture']}: {r['response']}" for r in responses])
        verdict = model(
            f"Aggregate these culturally-informed answers into one comprehensive and culturally respectful answer:\n{summary}"
        )
        return {
            **updated_state,
            "response_state": {
                **state.get("response_state", {}),
                "judged": verdict
            },
            "activate_compose": True,
            "__next__": "compose"
        }
    return updated_state


def route_to_cultures(state: GraphState, lambda_1: float = 0.6, lambda_2: float = 0.4, top_k: int = 3) -> Dict:
    q = state["question_meta"]["original"]
    d_u = get_text_embedding(str(state.get("user_profile", "")))
    sensitive_topics = state["question_meta"].get("sensitive_topics", [])

    topic_vectors = [get_text_embedding(topic) for topic in sensitive_topics]
    if not topic_vectors:
        topic_vectors = [get_text_embedding("general")]
    D_T = np.stack(topic_vectors)

    d_T_bar = np.mean(D_T, axis=0)
    d_TU = (lambda_1 * d_T_bar + lambda_2 * d_u) / (lambda_1 + lambda_2)

    culture_list = list(expert_classes.keys())
    D_E = np.stack([get_text_embedding(culture) for culture in culture_list])
    S = -np.sum(np.abs(D_E - d_TU), axis=1)

    k = min(top_k, len(culture_list))
    topk_indices = np.argsort(S)[-k:][::-1]
    S_k = S[topk_indices]
    s_max = np.max(S_k)
    exp_scores = np.exp(S_k - s_max)
    softmax_weights = exp_scores / np.sum(exp_scores)

    A = []
    for idx, weight in zip(topk_indices, softmax_weights):
        culture = culture_list[idx]
        P_j = [f"As a representative of {culture}, please consider:"]
        prompt_j = gen_prompt(q, d_T_bar, P_j)
        A.append((culture, weight, prompt_j))

    relevant_cultures = [culture for culture, _, _ in A]
    return {
        "question_meta": {
            **state["question_meta"],
            "relevant_cultures": relevant_cultures
        },
        "current_state": "router",
        "expert_weights_and_prompts": A
    }

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



def compose_final_response(state: GraphState) -> Dict:
    final = f"Culturally informed response: {state['response_state']['judged']}"
    return {
        "response_state": {**state.get("response_state", {}), "final": final},
        "current_state": "compose"
    }

# === Router Function ===
def analyzer_router(state: GraphState) -> List[str]:
    cultures = state["question_meta"].get("relevant_cultures", [])
    return [f"expert_{culture}" for culture in cultures]
