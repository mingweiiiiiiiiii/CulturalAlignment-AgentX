#!/usr/bin/env python3
"""
Test parallel optimization for expert queries
"""
import time
import json

# Monkey patch to use Ollama
import llmagentsetting.llm_clients as llm_clients
from llmagentsetting.ollama_client import OllamaClient

class OllamaAPIClient(OllamaClient):
    def __init__(self, state=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state

llm_clients.LambdaAPIClient = OllamaAPIClient

from compose_agent_optimized import compose_final_response_optimized
from node.compose_agent import compose_final_response

def test_parallel_vs_sequential():
    print("=== Testing Parallel vs Sequential Expert Queries ===\n")
    
    # Create test state
    state = {
        "activate_set": [
            {"culture": "China", "weight": 0.45},
            {"culture": "United States", "weight": 0.35},
            {"culture": "India", "weight": 0.20}
        ],
        "question_meta": {
            "original": "Should elderly parents move in with their adult children?",
            "sensitive_topics": ["Family care"],
            "relevant_cultures": ["China", "United States", "India"]
        },
        "response_state": {
            "expert_responses": []
        },
        "user_profile": {
            "age": "35",
            "ancestry": "Chinese"
        }
    }
    
    # Test sequential (original)
    print("1. Testing SEQUENTIAL expert queries...")
    state_seq = state.copy()
    state_seq["response_state"] = {"expert_responses": []}
    
    start = time.time()
    result_seq = compose_final_response(state_seq)
    seq_time = time.time() - start
    
    print(f"\nSequential total time: {seq_time:.2f}s")
    print(f"Number of responses: {len(state_seq['response_state'].get('expert_responses', []))}")
    
    # Test parallel (optimized)
    print("\n2. Testing PARALLEL expert queries...")
    state_par = state.copy()
    state_par["response_state"] = {"expert_responses": []}
    
    start = time.time()
    result_par = compose_final_response_optimized(state_par)
    par_time = time.time() - start
    
    print(f"\nParallel total time: {par_time:.2f}s")
    print(f"Number of responses: {len(state_par['response_state'].get('expert_responses', []))}")
    
    # Compare results
    print(f"\n=== Performance Comparison ===")
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Parallel: {par_time:.2f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")
    print(f"Time saved: {seq_time - par_time:.2f}s")

if __name__ == "__main__":
    test_parallel_vs_sequential()