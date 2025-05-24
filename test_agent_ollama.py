#!/usr/bin/env python3
"""
Test script to run the cultural agent using Ollama
"""
import json
from mylanggraph.graph import create_cultural_graph
from mylanggraph.types import GraphState
from utility.inputData import PersonaSampler

def test_agent_with_ollama():
    """Run a simple test of the agent using Ollama"""
    
    # Create graph
    graph = create_cultural_graph()
    
    # Create a sample persona and question
    sampler = PersonaSampler()
    profiles = sampler.sample_profiles(1)
    question, options = sampler.sample_question()
    
    merged_question = f"{question}\n\nOptions:\n" + \
        "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
    
    print(f"Test Profile: {json.dumps(profiles[0], indent=2)}")
    print(f"\nTest Question: {merged_question}")
    
    # Create initial state
    state: GraphState = {
        "user_profile": profiles[0],
        "question_meta": {
            "original": merged_question,
            "options": options,
            "sensitive_topics": [],
            "relevant_cultures": [],
        },
        "response_state": {
            "expert_responses": [],
        },
        "full_history": [],
        "planner_counter": 0,
        "activate_sensitivity_check": True,
        "activate_extract_topics": False,  # Disable to avoid Lambda API
        "activate_router": False,
        "activate_judge": False,
        "activate_compose": False,
        "current_state": "planner",
    }
    
    print("\n--- Running Agent ---")
    try:
        result = graph.invoke(state, config={
            "recursion_limit": 50,
            "configurable": {"thread_id": "test"},
            "verbose": True,
        })
        
        print("\n--- Result ---")
        print(f"Final Response: {result.get('response_state', {}).get('final', 'No response generated')}")
        
        return result
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_agent_with_ollama()