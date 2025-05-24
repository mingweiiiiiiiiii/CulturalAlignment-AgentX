import os
import sqlite3
from typing import List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from mylanggraph.custom_types import GraphState
from node.combined_analysis_cached import analyze_question_sensitivity_cached
from node.router_optimized import route_to_cultures_optimized, precompute_centroids
from node.compose_agent_cached import compose_final_response_cached
from node.planner_agent_node import planner_agent
from typing import Dict

def create_optimized_cultural_graph(cultures: Optional[List[str]] = None):
    """
    Creates an optimized cultural graph with:
    - Combined sensitivity + topic extraction
    - Parallel expert queries
    - Caching for all expensive operations
    - Pre-computed centroids for routing
    
    The optimized flow:
    1. Planner → Combined Analysis (sensitivity + topics)
    2. If sensitive → Router → Parallel Expert Composition
    3. If not sensitive → Direct to Composition (no experts)
    """
    cultures = cultures or ["United States", "China", "India", "Japan", "Turkey", "Vietnam"]
    
    # Pre-compute centroids at graph creation time
    print("Initializing optimized cultural graph...")
    precompute_centroids()
    
    # Build the graph
    builder = StateGraph(GraphState)
    
    # Add nodes - using optimized versions
    builder.add_node("planner", planner_optimized)
    builder.add_node("analyze", analyze_question_sensitivity_cached)
    builder.add_node("router", route_to_cultures_optimized)
    builder.add_node("compose", compose_final_response_cached)
    
    # Set entry point
    builder.set_entry_point("planner")
    
    # Define edges
    builder.add_conditional_edges(
        "planner",
        lambda state: [state["__next__"]] if "__next__" in state else [],
        ["analyze", "router", "compose"],
    )
    
    # After analysis, go back to planner for routing decision
    builder.add_edge("analyze", "planner")
    
    # After routing, go to compose
    builder.add_edge("router", "compose")
    
    # Compose is the end
    builder.add_edge("compose", END)
    
    # Save checkpoints
    os.makedirs("./data/graph_checkpoints", exist_ok=True)
    db_path = os.path.join(".", "data", "graph_checkpoints", "checkpoints_optimized.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Compile and return
    graph = builder.compile(checkpointer=memory).with_config(
        run_name="Optimized Cultural Analysis"
    )
    
    return graph

def planner_optimized(state) -> Dict:
    """
    Simplified planner for optimized workflow.
    Only needs to handle 3 states now:
    1. Initial → Combined analysis
    2. After analysis → Router (if sensitive) or Compose (if not)
    3. After router → Compose
    """
    from utility.measure_time import measure_time
    
    @measure_time
    def _planner_logic(state):
        counter = state.get("planner_counter", 0) + 1
        state["planner_counter"] = counter
        
        if counter == 1:
            # First step: always do combined analysis
            state["__next__"] = "analyze"
            
        elif counter == 2:
            # After analysis: check if sensitive
            if state.get("is_sensitive", False):
                state["__next__"] = "router"
            else:
                state["__next__"] = "compose"
                
        else:
            # Shouldn't reach here in optimized flow
            state["__next__"] = "compose"
        
        state["current_state"] = state["__next__"]
        return state
    
    return _planner_logic(state)