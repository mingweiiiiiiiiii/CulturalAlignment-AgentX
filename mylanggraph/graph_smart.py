"""
Smart cultural alignment graph with:
- Full 20 culture pool
- Top 5 expert selection
- Smart response generation (full only if culturally relevant)
"""
from typing import Optional, List, Dict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from node.sensitivity_optimized import analyze_question_sensitivity
from node.router_optimized_v2 import route_to_cultures_smart
from node.compose_agent_smart import compose_final_response_smart
from utility.measure_time import measure_time

def create_smart_cultural_graph():
    """
    Create optimized graph with smart cultural expert selection.
    Features:
    - 20 culture pool
    - Max 5 experts selected
    - Full responses only for culturally relevant questions
    - Brief inputs from less relevant cultures
    """
    print("Initializing smart cultural graph with 20 culture pool...")
    
    # Define workflow
    builder = StateGraph(dict)
    
    # Add nodes
    builder.add_node("analyze_sensitivity", analyze_question_sensitivity)
    builder.add_node("route_cultures", route_to_cultures_smart) 
    builder.add_node("compose_response", compose_final_response_smart)
    
    # Simple planner logic
    @measure_time
    def planner_smart(state: Dict) -> Dict:
        """Smart planner that routes based on sensitivity."""
        # Track steps
        if "steps" not in state:
            state["steps"] = []
        
        # Check if sensitive
        if state.get("is_sensitive", False):
            # Sensitive - route to cultural experts
            state["__next__"] = "route_cultures"
            state["steps"].append("Routing to cultural experts (sensitive question)")
        else:
            # Not sensitive - direct response
            state["__next__"] = "compose_response"
            state["steps"].append("Direct response (non-sensitive question)")
        
        return state
    
    builder.add_node("planner", planner_smart)
    
    # Set entry point
    builder.set_entry_point("analyze_sensitivity")
    
    # Add edges
    builder.add_edge("analyze_sensitivity", "planner")
    
    # Conditional routing from planner
    def route_from_planner(state: Dict) -> str:
        return state.get("__next__", "compose_response")
    
    builder.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "route_cultures": "route_cultures",
            "compose_response": "compose_response"
        }
    )
    
    # Final edges
    builder.add_edge("route_cultures", "compose_response")
    builder.set_finish_point("compose_response")
    
    # Compile with memory
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    print("Smart cultural graph ready!")
    print("- Pool: 20 diverse cultures")
    print("- Selection: Top 5 most relevant")
    print("- Responses: Full if relevant, brief if not")
    
    return graph

# Example usage
if __name__ == "__main__":
    import json
    from datetime import datetime
    
    # Create graph
    graph = create_smart_cultural_graph()
    
    # Test with a culturally sensitive question
    test_state = {
        "question_meta": {
            "original": "Should elderly parents live with their adult children?",
            "timestamp": datetime.now().isoformat()
        },
        "user_profile": {
            "age": 35,
            "location": "China",
            "cultural_background": "East Asian",
            "values": ["family harmony", "filial piety", "tradition"]
        },
        "steps": []
    }
    
    print("\nTesting smart cultural graph...")
    print(f"Question: {test_state['question_meta']['original']}")
    print(f"User: {test_state['user_profile']['location']}")
    
    # Run workflow
    config = {"configurable": {"thread_id": "smart_test_001"}}
    result = graph.invoke(test_state, config=config)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Sensitivity
    meta = result.get("question_meta", {})
    print(f"\nSensitivity Analysis:")
    print(f"  Score: {meta.get('sensitivity_score', 'N/A')}/10")
    print(f"  Is Sensitive: {meta.get('is_sensitive', 'N/A')}")
    
    # Expert selection
    if result.get("selected_cultures"):
        print(f"\nSelected Experts: {', '.join(result['selected_cultures'])}")
        
        # Response breakdown
        expert_responses = result.get("expert_responses", {})
        full_count = sum(1 for r in expert_responses.values() if r['response_type'] == 'full')
        brief_count = len(expert_responses) - full_count
        
        print(f"\nResponse Breakdown:")
        print(f"  Full responses: {full_count}")
        print(f"  Brief responses: {brief_count}")
        
        # Show which cultures gave full responses
        full_cultures = [c for c, r in expert_responses.items() if r['response_type'] == 'full']
        if full_cultures:
            print(f"  Full responses from: {', '.join(full_cultures)}")
    
    # Final response
    final = result.get("final_response", {})
    if final:
        print(f"\nFinal Response Preview:")
        print(f"{final.get('main_response', 'N/A')[:200]}...")
        
        if final.get('cultural_insights'):
            print(f"\nCultural Insights:")
            for insight in final['cultural_insights']:
                print(f"  • {insight}")
    
    # Workflow steps
    print(f"\nWorkflow Steps:")
    for step in result.get('steps', []):
        print(f"  - {step}")