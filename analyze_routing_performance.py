#!/usr/bin/env python3
"""
Analyze why cultural routing takes so long
"""
import time
import numpy as np
from node.cultural_expert_node import CulturalExpertManager
from node.embed_utils import embed_persona

def analyze_routing_performance():
    print("=== Analyzing Cultural Routing Performance ===\n")
    
    # Create a mock state
    state = {
        "user_profile": {
            "age": "35",
            "sex": "Male",
            "race": "Asian",
            "ancestry": "Chinese"
        }
    }
    
    # Initialize manager
    manager = CulturalExpertManager(state=state)
    
    # Test 1: Time expert generation
    print("1. Testing expert instance generation...")
    start = time.time()
    experts = manager.generate_expert_instances()
    expert_gen_time = time.time() - start
    print(f"   Generated {len(experts)} experts in {expert_gen_time:.3f}s")
    
    # Test 2: Time embedding generation for all experts
    print("\n2. Testing expert embedding generation...")
    start = time.time()
    expert_list, expert_embeddings = manager.get_all_persona_embeddings()
    embedding_time = time.time() - start
    print(f"   Generated embeddings for {len(expert_list)} experts in {embedding_time:.3f}s")
    print(f"   Average time per expert: {embedding_time/len(expert_list):.3f}s")
    
    # Test 3: Time a single embedding
    print("\n3. Testing single embedding generation...")
    test_persona = {"country": "test", "description": "test persona"}
    start = time.time()
    single_embedding = embed_persona(test_persona)
    single_time = time.time() - start
    print(f"   Single embedding took: {single_time:.3f}s")
    
    # Test 4: Analyze the embedding calls in routing
    print("\n4. Analyzing embedding calls in routing:")
    print(f"   - User profile embedding: 1 call")
    print(f"   - Expert embeddings: {len(expert_list)} calls (if not cached)")
    print(f"   - Topic embeddings: 1+ calls (depends on sensitive topics)")
    
    total_estimated = single_time * (1 + len(expert_list) + 1)
    print(f"\n   Estimated total time: {total_estimated:.3f}s")
    
    # Test 5: Check if caching works
    print("\n5. Testing caching...")
    start = time.time()
    expert_list2, expert_embeddings2 = manager.get_all_persona_embeddings()
    cached_time = time.time() - start
    print(f"   Cached retrieval took: {cached_time:.3f}s")
    print(f"   Cache working: {cached_time < 0.001}")
    
    return {
        "num_experts": len(expert_list),
        "expert_generation_time": expert_gen_time,
        "total_embedding_time": embedding_time,
        "single_embedding_time": single_time,
        "estimated_total": total_estimated
    }

if __name__ == "__main__":
    results = analyze_routing_performance()