from langgraph.graph import StateGraph, END
from custom_types import GraphState
from node import (
    planner_agent,
    determine_cultural_sensitivity,
    extract_sensitive_topics,
    route_to_cultures,
    compose_final_response,
    USExpert,
    ChineseExpert,
    IndianExpert,
)

# === Culture Definitions ===
DEFAULT_EXPERTS = {
    "US": USExpert(),
    "China": ChineseExpert(),
    "India": IndianExpert(),
}

def create_cultural_graph(cultures: Optional[List[str]] = None):
    """
    Creates a cultural graph for analyzing cultural sensitivities across different cultures.

    Args:
        cultures (list[str], optional): List of culture codes to include. Defaults to ["US", "China", "India"].

    Returns:
        CompiledGraph: A compiled LangGraph ready for execution.
    """
    cultures = cultures or ["US", "China", "India"]
    builder = StateGraph(GraphState)
    builder.set_entry_point("planner")

    # Core nodes
    builder.add_node("planner", planner_agent)
    builder.add_node("sensitivity_check", determine_cultural_sensitivity)
    builder.add_node("extract_topics", extract_sensitive_topics)
    builder.add_node("router", route_to_cultures)
    builder.add_node("compose", compose_final_response)

    # Cultural expert nodes
    for culture in cultures:
        if culture not in DEFAULT_EXPERTS:
            raise ValueError(f"No expert implementation for culture: {culture}")
        builder.add_node(f"expert_{culture}", DEFAULT_EXPERTS[culture])

    # Graph transitions
    builder.add_conditional_edges(
        "planner",
        lambda state: [state["__next__"]] if "__next__" in state else [],
        ["sensitivity_check", "router", "compose"]
    )

    builder.add_edge("sensitivity_check", "extract_topics")
    builder.add_edge("extract_topics", "router")

    builder.add_conditional_edges(
        "router",
        route_to_cultures, 
    )

    builder.add_edge("compose", END)

    return builder.compile()
