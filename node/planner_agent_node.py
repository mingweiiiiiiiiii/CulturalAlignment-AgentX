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


def planner_agent(state: GraphState, evaluator: EvaluationLosses) -> Dict:


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
