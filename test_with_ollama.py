#!/usr/bin/env python3
"""
Modified main.py to use Ollama instead of Lambda API
"""
import json
import math
import os
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

# Monkey patch to use Ollama instead of Lambda API
import llmagentsetting.llm_clients as llm_clients
from llmagentsetting.ollama_client import OllamaClient

# Replace LambdaAPIClient with OllamaClient
class OllamaAPIClient(OllamaClient):
    def __init__(self, state=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state

# Monkey patch the module
llm_clients.LambdaAPIClient = OllamaAPIClient

# Now import the rest
from mylanggraph.graph import create_cultural_graph
from mylanggraph.types import GraphState
from utility.inputData import PersonaSampler

def run_single_test():
    """Run a single test of the cultural alignment system with Ollama"""
    
    print("=== Testing Cultural Alignment with Ollama ===\n")
    
    # Create sampler and graph
    sampler = PersonaSampler()
    graph = create_cultural_graph()
    
    # Get a profile and question
    profiles = sampler.sample_profiles(1)
    profile = profiles[0]
    question, options = sampler.sample_question()
    
    merged_question = f"{question}\n\nOptions:\n" + \
        "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
    
    print(f"Profile: {json.dumps(profile, indent=2)}\n")
    print(f"Question: {merged_question}\n")
    
    # Create initial state
    state: GraphState = {
        "user_profile": profile,
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
        "node_times": {}
    }
    
    print("--- Running Cultural Analysis ---")
    start_time = time.time()
    
    try:
        result = graph.invoke(state, config={
            "recursion_limit": 50,
            "configurable": {"thread_id": "test"},
            "verbose": True,
        })
        
        elapsed = time.time() - start_time
        print(f"\n--- Analysis Complete ({elapsed:.2f}s) ---\n")
        
        # Show results
        meta = result.get("question_meta", {})
        print(f"Cultural Sensitivity:")
        print(f"- Is Sensitive: {meta.get('is_sensitive', False)}")
        print(f"- Score: {meta.get('sensitivity_score', 0)}/10")
        print(f"- Topics: {meta.get('sensitive_topics', [])}")
        print(f"- Cultures: {meta.get('relevant_cultures', [])}")
        
        # Show expert responses if any
        experts = result.get("response_state", {}).get("expert_responses", [])
        if experts:
            print(f"\n--- Expert Responses ({len(experts)}) ---")
            for i, expert in enumerate(experts):
                print(f"\nExpert {i+1} ({expert.get('culture', 'Unknown')}):")
                resp = expert.get('response', 'No response')
                print(f"{resp[:150]}..." if len(resp) > 150 else resp)
        
        # Final response
        final = result.get("response_state", {}).get("final", "")
        if final:
            print(f"\n--- Final Response ---")
            print(final)
            
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_single_test()