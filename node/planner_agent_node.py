from typing import Dict

# === Global counter for planner ===
planner_counter = 0

def planner_agent(state) -> Dict:
    """Planner that routes based on current state and graph iteration."""

    counter = state["planner_counter"] + 1 if "planner_counter" in state else 1
    
    if counter == 1:
        return {
            "planner_counter": counter,
            "activate_sensitivity_check": True,
            "__next__": "sensitivity_check",
            "current_state": "sensitivity_check"
        }
    elif counter == 2 and state.get("is_sensitive") is True:
        return {
            "planner_counter": counter,
            "activate_extract_topics": True,
            "__next__": "extract_topics",
            "current_state": "extract_topics"
        }
    elif counter == 2 and not state.get("is_sensitive", False):
        return {
            "planner_counter": counter,
            "__next__": "compose",
            "current_state": "compose"
        }
    elif counter >= 3:
        if state.get("topics_extracted"):
            return {
                "planner_counter": counter,
                "activate_router": True,
                "__next__": "router",
                "current_state": "router"
            }
        else:
            return {
                "planner_counter": counter,
                "__next__": "compose",
                "current_state": "compose"
            }