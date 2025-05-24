#!/usr/bin/env python3
import requests
import json
import time

# Test both APIs for comparison
optimized_url = "http://localhost:8002/api/cultural-workflow-optimized"

# Simple test data - less sensitive question
test_data = {
    "persona": {
        "age": "35",
        "race": "Asian",
        "sex": "Female",
        "ancestry": "Chinese",
        "country": "United States"
    },
    "question": "What are your thoughts on work-life balance?",
    "options": [
        "Work should always come first",
        "Personal life is more important",
        "Both are equally important",
        "It depends on life circumstances"
    ],
    "cultures": ["United States", "China", "Japan"]  # Just 3 cultures
}

print("=" * 80)
print("CACHE EFFECTIVENESS DEMONSTRATION")
print("=" * 80)
print(f"Question: {test_data['question']}")
print(f"Testing with {len(test_data['cultures'])} cultural experts")
print("-" * 80)

# First call - cold cache
print("\n🧊 First call (COLD cache)...")
try:
    start_time = time.time()
    response = requests.post(optimized_url, json=test_data, timeout=120)
    first_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Completed in {result['processing_time']:.1f} seconds")
        print(f"\n💾 Cache Statistics:")
        for cache_type, stats in result.get('cache_stats', {}).items():
            print(f"   - {cache_type}: {stats['hit_rate']} ({stats['misses']} misses)")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text[:200])
except Exception as e:
    print(f"❌ Error: {e}")

# Second call - warm cache
print("\n" + "-" * 80)
print("\n🔥 Second call (WARM cache)...")
try:
    start_time = time.time()
    response = requests.post(optimized_url, json=test_data, timeout=120)
    second_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Completed in {result['processing_time']:.1f} seconds")
        print(f"\n💾 Cache Statistics:")
        for cache_type, stats in result.get('cache_stats', {}).items():
            print(f"   - {cache_type}: {stats['hit_rate']} ({stats['hits']} hits)")
        
        print(f"\n🚀 Performance Improvement:")
        if first_time > 0:
            speedup = first_time / result['processing_time']
            print(f"   - Speedup: {speedup:.1f}x faster")
            print(f"   - Time saved: {first_time - result['processing_time']:.1f} seconds")
    else:
        print(f"❌ Error: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Third call with slightly different persona - partial cache hit
print("\n" + "-" * 80)
print("\n🌡️  Third call (different persona, same question)...")
test_data["persona"]["age"] = "45"  # Change age
try:
    start_time = time.time()
    response = requests.post(optimized_url, json=test_data, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Completed in {result['processing_time']:.1f} seconds")
        print(f"\n💾 Cache Statistics:")
        for cache_type, stats in result.get('cache_stats', {}).items():
            print(f"   - {cache_type}: {stats['hit_rate']} (partial cache benefits)")
    else:
        print(f"❌ Error: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)