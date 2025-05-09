import os
import sqlite3
from typing import List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from mylanggraph.custom_types import GraphState
from node.compose_agent import compose_final_response
from node.planner_agent_node import planner_agent
from node.router_node import route_to_cultures
from node.sen_agent_node import determine_cultural_sensitivity
from node.extract_topics_agent_node import extract_sensitive_topics


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

    # Core nodes
    builder.add_node("planner", planner_agent)
    builder.add_node("sensitivity_check", determine_cultural_sensitivity)
    builder.add_node("extract_topics", extract_sensitive_topics)
    builder.add_node("router", route_to_cultures)
    builder.add_node("compose", compose_final_response)
    builder.set_entry_point("planner")
    # Graph transitions
    builder.add_conditional_edges(
        "planner",
        lambda state: [state["__next__"]] if "__next__" in state else [],
        ["sensitivity_check", "extract_topics", "router", "compose"],
    )
    #builder.add_edge("sensitivity_check", "extract_topics")
    #builder.add_edge("extract_topics", "router")
    #builder.add_edge("router", "planner")
    #builder.add_edge("planner", "compose")

    # These bring control back to planner after each node
    builder.add_edge("sensitivity_check", "planner")
    builder.add_edge("extract_topics", "planner")
    builder.add_edge("router", "planner")
    builder.add_edge("compose", END)

    # Save checkpoints
    os.makedirs("./data/graph_checkpoints", exist_ok=True)
    db_path = os.path.join(
        ".", "data", "graph_checkpoints", "checkpoints.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    # --- The missing part: compile and return the graph ---
    graph = builder.compile(checkpointer=memory).with_config(
        run_name="Starting running"
    )
    return graph
