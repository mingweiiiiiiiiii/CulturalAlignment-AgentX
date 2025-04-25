import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from typing import Dict, List
from node.types import GraphState
from sklearn.metrics.pairwise import cosine_similarity

# === Global counter for planner ===
# This counter is used to track the number of planning iterations in the cultural analysis process.
planner_counter = 0


# === Load BERT model and tokenizer once ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()


def extract_sensitive_topics(state: GraphState, evaluator: EvaluationLosses) -> Dict:
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
