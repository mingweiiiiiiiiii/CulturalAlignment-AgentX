from typing import Optional, List  
from langgraph.graph import StateGraph, END
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from custom_types import GraphState
from node import (
    planner_agent,
    determine_cultural_sensitivity,
    extract_sensitive_topics,
    route_to_cultures,
    compose_final_response,
)

# You need to make sure DEFAULT_EXPERTS is imported or defined too
from experts import DEFAULT_EXPERTS  # <- if it’s in another file

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


    # Graph transitions
    builder.add_conditional_edges(
        "planner",
        lambda state: [state["__next__"]] if "__next__" in state else [],
        ["sensitivity_check", "router", "compose"]
    )
    builder.add_edge("sensitivity_check", "extract_topics")
    builder.add_edge("extract_topics", "router")
    builder.add_edge("router", "planner")
    builder.add_edge("planner", "compose")
    builder.add_edge("compose", END)

    # Save checkpoints
    os.makedirs("./data/graph_checkpoints", exist_ok=True)
    db_path = os.path.join(".", "data", "graph_checkpoints", "checkpoints.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    # --- The missing part: compile and return the graph ---
    graph = builder.compile(checkpointer=memory)
    return graph
