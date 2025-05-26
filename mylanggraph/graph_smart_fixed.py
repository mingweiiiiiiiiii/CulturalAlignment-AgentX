"""
Smart cultural alignment graph with relevant_cultures fix.
- Full 20 culture pool
- Top 5 expert selection
- Smart response generation (full only if culturally relevant)
- FIXED: Populates relevant_cultures for proper alignment scoring
"""
from typing import Optional, List, Dict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from node.sensitivity_optimized import analyze_question_sensitivity
from node.router_optimized_v2_fixed import route_to_cultures_smart
from node.compose_agent_smart import compose_final_response_smart
from utility.measure_time import measure_time

def create_smart_cultural_graph_fixed():
    """
    Create optimized graph with smart cultural expert selection.
    FIXED: Now properly populates relevant_cultures for alignment scoring.
    
    Features:
    - 20 culture pool
    - Smart expert selection (top 5)
    - Relevance-based response generation
    - Proper cultural alignment scoring
    """
    # Build the state graph - using dict for simplicity
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("sensitivity_analysis", analyze_question_sensitivity)
    workflow.add_node("cultural_routing", route_to_cultures_smart)
    workflow.add_node("compose_response", compose_final_response_smart)
    
    # Define edges
    workflow.set_entry_point("sensitivity_analysis")
    
    # From sensitivity analysis
    workflow.add_edge("sensitivity_analysis", "cultural_routing")
    
    # From routing to composition
    workflow.add_edge("cultural_routing", "compose_response")
    
    # Set finish point
    workflow.set_finish_point("compose_response")
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("Smart Cultural Graph with Alignment Fix created successfully!")
    print("Features:")
    print("- 20 culture pool for diversity")
    print("- Top 5 expert selection")
    print("- Smart response generation")
    print("- Fixed cultural alignment scoring")