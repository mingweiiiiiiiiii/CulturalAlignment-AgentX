import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from typing import Dict, List
from .types import GraphState
from sklearn.metrics.pairwise import cosine_similarity

# === Global counter for planner ===
# This counter is used to track the number of planning iterations in the cultural analysis process.
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


def route_to_cultures(
    state: GraphState,
    expert_list: List[str],
    expert_embeddings: np.ndarray,
    prompt_libraries: Dict[str, List[str]],
    lambda_1: float = 0.6,
    lambda_2: float = 0.4,
    top_k: int = 3,
    tau: float = -30.0
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




def compose_final_response(
    state: GraphState,
    activate_set: List[Tuple[str, float, str]],  # (culture, weight, prompt_j)
    top_n: int = 3
) -> Dict:

    user_profile = state.get("user_profile", {})
    question_meta = state.get("question_meta", {})
    preferences = user_profile.get("preferences", {})
    demographics = user_profile.get("demographics", {})

    # Step 1: Select top-N cultures by weight
    top_cultures = sorted(activate_set, key=lambda x: x[1], reverse=True)[:top_n]
    expert_responses: List[ExpertResponse] = [
        {"culture": culture, "response": prompt} for culture, _, prompt in top_cultures
    ]

    # Step 2: Create an LLM prompt
    question = question_meta.get("original", "")
    sensitive_topics = question_meta.get("sensitive_topics", [])
    relevant_cultures = question_meta.get("relevant_cultures", [])

    prompt_parts = [
        "You are a culturally-aware assistant tasked with composing a final response that is sensitive to the user's background and preferences.",
        f"Question: {question}",
        "\nUser Profile:",
        f"- Demographics: {demographics}",
        f"- Preferences: {preferences}",
        "\nCulturally Diverse Expert Responses:"
    ]

    for i, resp in enumerate(expert_responses, 1):
        prompt_parts.append(f"{i}. ({resp['culture']}) {resp['response']}")

    if sensitive_topics:
        prompt_parts.append(f"\nSensitive Topics: {', '.join(sensitive_topics)}")
    if relevant_cultures:
        prompt_parts.append(f"Relevant Cultures: {', '.join(relevant_cultures)}")

    prompt_parts.append("\nPlease write a final, thoughtful, and culturally-informed response that blends the above insights in a coherent and respectful way.")

    llm_prompt = "\n".join(prompt_parts)

    # Step 3: Generate final composed response
    final_response = model(llm_prompt)

    # Step 4: Update response state
    response_state = state.get("response_state", {})
    response_state.update({
        "expert_responses": expert_responses,
        "final": final_response
    })

    return {
        "response_state": response_state,
        "current_state": "compose"
    }
# === Router Function ===
def analyzer_router(state: GraphState) -> List[str]:
    cultures = state["question_meta"].get("relevant_cultures", [])
    return [f"expert_{culture}" for culture in cultures]
