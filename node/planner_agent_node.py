
import numpy as np
import re
from typing import Dict, List

from sklearn.metrics.pairwise import cosine_similarity

# === Global counter for planner ===
# This counter is used to track the number of planning iterations in the cultural analysis process.
planner_counter = 0



def planner_agent(state) -> Dict:


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