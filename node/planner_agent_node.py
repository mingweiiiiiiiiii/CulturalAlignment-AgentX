from typing import Dict

# === Global counter for planner ===
planner_counter = 0


def planner_agent(state) -> Dict:
    """Planner that routes based on current state and graph iteration."""

    global planner_counter
    planner_counter += 1
    counter = planner_counter
    updated_state = {"current_state": "planner", "planner_counter": counter}

    if counter == 1:
        # First step: Check sensitivity
        return {
            **updated_state,
            "activate_sensitivity_check": True,
            "__next__": "sensitivity_check",
        }
    elif counter == 2 and state.get("is_sensitive") is True:
        # Second step: Extract sensitive topics
        return {
            **updated_state,
            "activate_extract_topics": True,
            "__next__": "extract_topics",
        }
    elif counter == 2 and state.get("is_sensitive") is False:
        # If not sensitive, go directly to composing
        return {
            **updated_state,
            "__next__": "compose",
        }
    elif counter >= 3:
        # After topic extraction and memory hook, route to cultures
        if state.get("topics_extracted"):
            return {
                **updated_state,
                "activate_router": True,
                "__next__": "router",
            }
        else:
            # Fallback: If no topics extracted, compose response anyway
            return {
                **updated_state,
                "__next__": "compose",
            }
