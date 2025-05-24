"""
Simple test of main.py functionality without full benchmark.
"""
import json
import time
from datetime import datetime
from mylanggraph.graph import create_cultural_graph
from mylanggraph.types import GraphState
from utility.baseline import generate_baseline_essay
from utility.inputData import PersonaSampler

def run_single_test():
    """Run a single test with the cultural agent."""
    print("=" * 80)
    print("CULTURAL AGENT - SINGLE TEST RUN")
    print("=" * 80)
    
    # Initialize
    sampler = PersonaSampler()
    graph = create_cultural_graph()
    
    # Sample one profile and question
    profiles = sampler.sample_profiles(1)
    question, options = sampler.sample_question()
    merged_question = f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
    
    print(f"\nUser Profile:")
    print(json.dumps(profiles[0], indent=2))
    print(f"\nQuestion: {merged_question}")
    print("-" * 80)
    
    # Create state
    state = {
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
        "activate_extract_topics": True,
        "activate_router": False,
        "activate_judge": False,
        "activate_compose": False,
        "current_state": "planner",
    }
    
    # Run model
    print("\nRunning cultural agent...")
    model_start = time.perf_counter()
    
    try:
        result = graph.invoke(state, config={
            "recursion_limit": 200,
            "configurable": {"thread_id": "test_001"},
            "verbose": True,
        })
        model_end = time.perf_counter()
        model_latency = model_end - model_start
        
        print(f"\n✅ Model completed in {model_latency:.2f} seconds")
        
        # Extract results
        final_response = result.get("response_state", {}).get("final", "No response generated")
        expert_responses = result.get("response_state", {}).get("expert_responses", [])
        
        print(f"\nExpert Consultations: {len(expert_responses)} experts")
        for resp in expert_responses[:3]:  # Show first 3
            culture = resp.get("culture", "Unknown")
            response_preview = resp.get("response", "")[:100] + "..."
            print(f"  - {culture}: {response_preview}")
        
        print(f"\nFinal Response:")
        print("-" * 40)
        print(final_response)
        
    except Exception as e:
        print(f"\n❌ Error running model: {e}")
        import traceback
        traceback.print_exc()
    
    # Run baseline for comparison
    print("\n" + "-" * 80)
    print("Running baseline...")
    baseline_start = time.perf_counter()
    
    try:
        essay = generate_baseline_essay(profiles, merged_question)
        baseline_end = time.perf_counter()
        baseline_latency = baseline_end - baseline_start
        
        print(f"\n✅ Baseline completed in {baseline_latency:.2f} seconds")
        print(f"\nBaseline Response:")
        print("-" * 40)
        print(essay)
        
    except Exception as e:
        print(f"\n❌ Error running baseline: {e}")

if __name__ == "__main__":
    run_single_test()