from langgraph.graph import StateGraph, END
from .types import GraphState
from .nodes import (
    planner_agent,
    determine_cultural_sensitivity,
    extract_sensitive_topics,
    route_to_cultures,
    judge_agent,
    compose_final_response,
    cultural_expert_node_factory,
    analyzer_router
)
from .db import database_node

def create_cultural_graph(cultures=None):
    """Creates a cultural graph for analyzing cultural sensitivities across different cultures.

    This function builds a directed graph of nodes representing different agents and processing steps
    for cultural analysis. The graph includes nodes for planning, sensitivity checking, topic extraction,
    cultural routing, expert analysis, and response composition.

    Args:
        cultures (list[str], optional): List of culture codes to include in the graph.
            Defaults to ["US", "China", "India"].

    Returns:
        CompiledGraph: A compiled graph object ready for execution with the following key nodes:
            - planner: Entry point for graph execution
            - sensitivity_check: Determines cultural sensitivity of input
            - extract_topics: Extracts potentially sensitive topics
            - router: Routes analysis to appropriate cultural experts
            - expert_{culture}: Cultural expert nodes for each specified culture
            - judge: Evaluates cultural expert feedback
            - compose: Composes final response
            - database: Handles data storage and retrieval

    Example:
        >>> graph = create_cultural_graph(["US", "Japan", "India"])
        >>> graph.run({"input": "some text to analyze"})
    """

    if cultures is None:
        cultures = ["US", "China", "India"]

    builder = StateGraph(GraphState)
    builder.set_entry_point("planner")

    builder.add_node("planner", planner_agent)
    builder.add_node("sensitivity_check", determine_cultural_sensitivity)
    builder.add_node("extract_topics", extract_sensitive_topics)
    builder.add_node("router", route_to_cultures)
    builder.add_node("compose", compose_final_response)
    builder.add_node("database", lambda state: {"db_result": None, "current_state": "database"})

    for culture in cultures:
        builder.add_node(f"expert_{culture}", cultural_expert_node_factory(culture))

    builder.add_conditional_edges(
        "planner",
        lambda state: [state["__next__"]] if "__next__" in state else [],
        ["sensitivity_check", "router", "compose", "database"]
    )

    builder.add_edge("sensitivity_check", "extract_topics")
    builder.add_edge("extract_topics", "database")
    builder.add_conditional_edges("database", lambda state: ["planner"], ["planner"])
    builder.add_conditional_edges("router", analyzer_router, [f"expert_{c}" for c in cultures])

    for culture in cultures:
        builder.add_edge(f"expert_{culture}", "planner")

    builder.add_edge("compose", END)

    return builder.compile()




