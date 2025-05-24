#!/usr/bin/env python3
import requests
import json
import time

# Test both APIs for comparison
original_url = "http://localhost:8000/api/cultural-workflow"
optimized_url = "http://localhost:8002/api/cultural-workflow-optimized"

# Test data - sensitive question
test_data = {
    "persona": {
        "age": "35",
        "race": "Asian",
        "sex": "Female",
        "ancestry": "Chinese",
        "country": "United States"
    },
    "question": "What is your opinion on government policies regarding individual freedoms versus collective security?",
    "options": [
        "Individual freedoms should always come first",
        "Security is more important than personal freedoms",
        "There must be a careful balance between both",
        "It depends on the specific cultural context"
    ],
    "cultures": ["United States", "China", "India", "Japan", "Turkey", "Vietnam"]
}

print("=" * 80)
print("CULTURAL ALIGNMENT OPTIMIZATION BENCHMARK")
print("=" * 80)
print(f"Question: {test_data['question']}")
print(f"Testing with {len(test_data['cultures'])} cultural experts")
print("-" * 80)

# Test optimized version first (cold start)
print("\n🚀 Testing OPTIMIZED workflow (cold start)...")
try:
    start_time = time.time()
    response = requests.post(optimized_url, json=test_data, timeout=120)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Completed in {result['processing_time']:.1f} seconds")
        print(f"\n📊 Optimization Metrics:")
        metrics = result.get('optimization_metrics', {})
        print(f"   - Parallel execution: {metrics.get('parallel_execution_time', 0):.1f}s")
        print(f"   - Est. sequential time: {metrics.get('estimated_sequential_time', 0):.1f}s")
        print(f"   - Time saved: {metrics.get('time_saved', 0):.1f}s")
        print(f"   - Speedup factor: {metrics.get('speedup_factor', 1):.1f}x")
        print(f"   - Total LLM calls: {metrics.get('total_llm_calls', 'N/A')}")
        
        print(f"\n💾 Cache Statistics (cold):")
        for cache_type, stats in result.get('cache_stats', {}).items():
            print(f"   - {cache_type}: {stats['hit_rate']} hit rate")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Error calling optimized API: {e}")

# Test again to see cache benefits
print("\n" + "-" * 80)
print("\n🚀 Testing OPTIMIZED workflow (with cache)...")
try:
    start_time = time.time()
    response = requests.post(optimized_url, json=test_data, timeout=120)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Completed in {result['processing_time']:.1f} seconds")
        
        print(f"\n💾 Cache Statistics (warm):")
        for cache_type, stats in result.get('cache_stats', {}).items():
            print(f"   - {cache_type}: {stats['hit_rate']} hit rate ({stats['hits']} hits)")
        
        print(f"\n📝 Expert Responses: {len(result['expert_responses'])} experts consulted")
        print(f"📝 Final response preview: {result['final_response'][:200]}...")
    else:
        print(f"❌ Error: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Compare with original
print("\n" + "-" * 80)
print("\n📊 Testing ORIGINAL workflow for comparison...")
try:
    start_time = time.time()
    response = requests.post(original_url, json=test_data, timeout=120)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Original completed in {result['processing_time']:.1f} seconds")
        
        print(f"\n⏱️  Node Timing Comparison:")
        for node, timing in result['node_times'].items():
            print(f"   - {node}: {timing:.1f}s")
    else:
        print(f"❌ Error: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)