import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from typing import Dict, List
from .types import GraphState
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
def determine_cultural_sensitivity(state: GraphState) -> Dict:
    question = state["question_meta"]["original"]
    prompt = (
        f"On a scale of 0 (not sensitive) to 10 (highly sensitive), rate the cultural sensitivity of the following question.\n"
        f"Consider potential for stereotyping, cultural offense, relevance to traditions, value systems, and identity.\n"
        f"Question: {question}\n"
        f"Return only a number."
    )
    response = model(prompt)
    try:
        sensitivity_score = int(re.search(r"\d+", response).group())
    except Exception:
        sensitivity_score = 0

    return {
        "question_meta": {
            **state["question_meta"],
            "is_sensitive": sensitivity_score >= 5,
            "sensitivity_score": sensitivity_score
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

  """Plans and orchestrates the flow of the conversation processing pipeline.

    This function manages the state transitions and data flow between different components
    of the system, including sensitivity checking, topic extraction, database operations,
    and response composition.

    Args:
        state (GraphState): Current state object containing conversation context and flags.

    Returns:
        Dict: Updated state dictionary with next action and relevant flags/data.
            Contains keys:
            - current_state (str): Always set to "planner"
            - planner_counter (int): Incremented counter tracking planner iterations
            - __next__ (str): Next node to execute
            - Various activation flags and data depending on the stage:
                - activate_sensitivity_check (bool)
                - activate_extract_topics (bool) 
                - activate_router (bool)
                - activate_judge (bool)
                - activate_compose (bool)
                - db_action (str)
                - db_key (str)
                - question_meta (dict)
                - response_state (dict)

    Flow:
        1. Initiates sensitivity check
        2. Triggers topic extraction
        3. Retrieves sensitive topics from database
        4. Routes the conversation
        5. Retrieves judged response from database
        6. Activates response composition
    """
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

def judge_agent(state: GraphState) -> Dict:
    responses = state["response_state"].get("expert_responses", [])
    summary = "\n".join([f"{r['culture']}: {r['response']}" for r in responses])
    verdict = model(f"Aggregate these culturally-informed answers into one comprehensive and culturally respectful answer:\n{summary}")
    return {
        "response_state": {**state.get("response_state", {}), "judged": verdict},
        "activate_judge": True,
        "db_action": "write",
        "db_key": "judged_response",
        "db_value": verdict,
        "__next__": "database",
        "current_state": "judge"
    }

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
