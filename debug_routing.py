#!/usr/bin/env python3
"""
Debug the routing performance in actual workflow
"""
import time
import json
import numpy as np

# Monkey patch to use Ollama
import llmagentsetting.llm_clients as llm_clients
from llmagentsetting.ollama_client import OllamaClient

class OllamaAPIClient(OllamaClient):
    def __init__(self, state=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state

llm_clients.LambdaAPIClient = OllamaAPIClient

from node.router_node import route_to_cultures
from node.cultural_expert_node import CulturalExpertManager
from node.embed_utils import embed_persona

def debug_routing():
    print("=== Debugging Routing Performance ===\n")
    
    # Create test state similar to workflow
    state = {
        "question_meta": {
            "original": "Should elderly parents move in with their adult children?",
            "sensitive_topics": ["Cultural Expectations of Family Care"],
            "relevant_cultures": [],
        },
        "user_profile": {
            "age": "35",
            "sex": "Male",
            "race": "Asian",
            "ancestry": "Chinese",
            "household_language": "Mandarin",
            "lifestyle": "Tech-focused urban professional",
            "ideology": "Moderate",
            "religion": "Buddhist"
        }
    }
    
    # Initialize manager to see how many experts
    manager = CulturalExpertManager(state=state)
    experts = manager.generate_expert_instances()
    print(f"Number of cultural experts: {len(experts)}")
    print(f"Expert countries: {list(experts.keys())}")
    
    # Time the actual routing function
    print("\n--- Timing route_to_cultures ---")
    start = time.time()
    result = route_to_cultures(state)
    total_time = time.time() - start
    
    print(f"\nTotal routing time: {total_time:.3f}s")
    print(f"Selected experts: {result}")
    
    # Break down the timing
    print("\n--- Breaking down operations ---")
    
    # Time user embedding
    start = time.time()
    user_embedding = embed_persona(state["user_profile"])
    user_embed_time = time.time() - start
    print(f"1. User profile embedding: {user_embed_time:.3f}s")
    
    # Time topic embeddings
    start = time.time()
    topic_embeddings = [
        embed_persona({"country": topic}) 
        for topic in state["question_meta"]["sensitive_topics"]
    ]
    topic_embed_time = time.time() - start
    print(f"2. Topic embeddings ({len(state['question_meta']['sensitive_topics'])} topics): {topic_embed_time:.3f}s")
    
    # Check if expert embeddings are cached
    manager2 = CulturalExpertManager(state=state)
    start = time.time()
    expert_list, expert_embeddings = manager2.get_all_persona_embeddings()
    expert_embed_time = time.time() - start
    print(f"3. Expert embeddings (first call): {expert_embed_time:.3f}s")
    
    # Test second call (should be cached)
    start = time.time()
    expert_list2, expert_embeddings2 = manager2.get_all_persona_embeddings()
    cached_time = time.time() - start
    print(f"4. Expert embeddings (cached): {cached_time:.3f}s")
    
    # The actual issue might be in the cultural expert responses
    print("\n--- Testing expert response generation ---")
    test_expert = experts["China"]
    start = time.time()
    try:
        response = test_expert.generate_response("Test question")
        expert_response_time = time.time() - start
        print(f"5. Single expert LLM response: {expert_response_time:.3f}s")
    except Exception as e:
        print(f"5. Expert response error: {e}")

if __name__ == "__main__":
    debug_routing()