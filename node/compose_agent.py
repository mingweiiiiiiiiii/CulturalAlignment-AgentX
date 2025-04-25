import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from typing import Dict, List
from node.types import GraphState
from sklearn.metrics.pairwise import cosine_similarity


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
