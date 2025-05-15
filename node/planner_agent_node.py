from typing import Dict
from utility.measure_time import measure_time

@measure_time
def planner_agent(state) -> Dict:
    counter = state.get("planner_counter", 0) + 1
    state["planner_counter"] = counter

    if counter == 1:
        state["activate_sensitivity_check"] = True
        state["__next__"] = "sensitivity_check"

    elif counter == 2 and state.get("is_sensitive", False):
        state["activate_extract_topics"] = True
        state["__next__"] = "extract_topics"

    elif counter == 2 and not state.get("is_sensitive", False):
        state["activate_compose"] = True
        state["__next__"] = "compose"

    elif counter == 3:
        if state.get("topics_extracted", False):
            state["activate_router"] = True
            state["__next__"] = "router"
        else:
            state["activate_compose"] = True
            state["__next__"] = "compose"

    elif counter >= 4:
        state["activate_compose"] = True
        state["__next__"] = "compose"

    state["current_state"] = state["__next__"]
    return state
