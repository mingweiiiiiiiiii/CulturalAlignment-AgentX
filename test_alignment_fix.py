#!/usr/bin/env python3
"""
Quick test script to verify cultural alignment scoring fix.
"""

import sys
import time
from mylanggraph.graph_smart import create_smart_cultural_graph

def test_alignment_fix():
    """Test the cultural alignment fix with a simple example."""
    print("Testing Cultural Alignment Fix")
    print("=" * 50)
    
    # Create graph
    graph = create_smart_cultural_graph()
    
    # Test state
    state = {
        "user_profile": {
            "place of birth": "California/CA",
            "ethnicity": "Mexican",
            "household language": "Spanish"
        },
        "question_meta": {
            "original": "Should children always obey their parents?",
            "options": ["Yes", "No", "Sometimes"],
            "sensitive_topics": [],
            "relevant_cultures": []
        },
        "steps": []
    }
    
    print(f"Question: {state['question_meta']['original']}")
    print(f"User profile: {state['user_profile']}")
    print()
    
    # Run the graph
    config = {"configurable": {"thread_id": "test_alignment"}}
    
    try:
        print("Running cultural alignment system...")
        start_time = time.perf_counter()
        result = graph.invoke(state, config=config)
        end_time = time.perf_counter()
        
        print(f"Completed in {end_time - start_time:.2f} seconds")
        print()
        
        # Check results
        print("Results:")
        print("-" * 30)
        
        # User's relevant cultures
        user_cultures = result.get("user_relevant_cultures", [])
        print(f"User's relevant cultures: {user_cultures}")
        
        # Selected experts
        selected_cultures = result.get("selected_cultures", [])
        print(f"Selected experts: {selected_cultures}")
        
        # Expert responses
        expert_responses = result.get("expert_responses", {})
        full_responses = [c for c, info in expert_responses.items() if info.get('response_type') == 'full']
        brief_responses = [c for c, info in expert_responses.items() if info.get('response_type') == 'brief']
        
        print(f"Full responses: {len(full_responses)} - {full_responses}")
        print(f"Brief responses: {len(brief_responses)} - {brief_responses}")
        
        # Calculate alignment manually
        if expert_responses and user_cultures:
            aligned_experts = [c for c in full_responses if c in user_cultures]
            if full_responses:
                alignment_score = len(aligned_experts) / len(full_responses)
                print(f"Cultural alignment score: {alignment_score:.2f}")
                print(f"Aligned experts: {aligned_experts}")
            else:
                print("Cultural alignment score: 0.00 (no full responses)")
        else:
            print("Cultural alignment score: 0.00 (no data)")
        
        # Check if fix worked
        if len(full_responses) > 0:
            print("\n✅ SUCCESS: Experts are providing full responses!")
            if any(c in user_cultures for c in full_responses):
                print("✅ SUCCESS: Some experts match user's cultural context!")
            else:
                print("⚠️  WARNING: No experts match user's cultural context")
        else:
            print("\n❌ ISSUE: No experts providing full responses")
            print("   This means the scoring fix didn't work")
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_alignment_fix()
