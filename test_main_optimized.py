"""
Test run for optimized cultural agent.
"""
import sys
sys.path.append('/app')

import json
import time
from datetime import datetime
from mylanggraph.graph_optimized import create_optimized_cultural_graph
from utility.inputData import PersonaSampler

def run_optimized_test():
    """Run a single test with the optimized cultural agent."""
    print("=" * 80)
    print("OPTIMIZED CULTURAL AGENT - TEST RUN")
    print("=" * 80)
    
    # Initialize
    sampler = PersonaSampler()
    graph = create_optimized_cultural_graph()
    
    # Sample one profile and question
    profiles = sampler.sample_profiles(1)
    question, options = sampler.sample_question()
    merged_question = f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
    
    print(f"\nUser Profile:")
    for key, value in profiles[0].items():
        if key in ['age', 'sex', 'race', 'location', 'ideology', 'political views']:
            print(f"  {key}: {value}")
    
    print(f"\nQuestion: {merged_question}")
    print("-" * 80)
    
    # Create state for optimized graph
    state = {
        "user_profile": profiles[0],
        "question_meta": {
            "original": merged_question,
            "options": options,
            "sensitive_topics": [],
            "relevant_cultures": [],
        },
        "is_sensitive": False,
        "activate_sensitivity_check": True,
        "activate_extract_topics": False,
        "activate_router": False,
        "activate_compose": False,
        "current_state": "sensitivity_check",
        "__start__": "sensitivity_check",
        "steps": [],
        "expert_consultations": {},
        "final_response": {},
        "planner_counter": 0,
        "response_state": {
            "expert_responses": [],
            "final": ""
        }
    }
    
    # Run optimized model
    print("\nRunning optimized cultural agent...")
    model_start = time.perf_counter()
    
    try:
        config = {"configurable": {"thread_id": "optimized_test_001"}}
        result = graph.invoke(state, config=config)
        model_end = time.perf_counter()
        model_latency = model_end - model_start
        
        print(f"\n✅ Optimized model completed in {model_latency:.2f} seconds")
        
        # Extract results
        meta = result.get("question_meta", {})
        print(f"\nSensitivity Analysis:")
        print(f"  - Score: {meta.get('sensitivity_score', 'N/A')}/10")
        print(f"  - Is Sensitive: {meta.get('is_sensitive', 'N/A')}")
        
        expert_responses = result.get("response_state", {}).get("expert_responses", [])
        print(f"\nExpert Consultations: {len(expert_responses)} experts")
        
        for resp in expert_responses[:3]:
            if isinstance(resp, dict):
                culture = resp.get("culture", "Unknown")
                response_text = resp.get("response", "")[:100] + "..."
                print(f"  - {culture}: {response_text}")
        
        final_response = result.get("response_state", {}).get("final", "No response")
        print(f"\nFinal Response:")
        print("-" * 40)
        print(final_response[:500] + "..." if len(final_response) > 500 else final_response)
        
        # Show cache statistics if available
        steps = result.get("steps", [])
        cache_steps = [s for s in steps if "Cache" in s]
        if cache_steps:
            print(f"\nCache Performance:")
            for step in cache_steps[:5]:
                print(f"  - {step}")
        
    except Exception as e:
        print(f"\n❌ Error running optimized model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_optimized_test()