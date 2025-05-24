#!/usr/bin/env python3
import requests
import json
import time

# API endpoint
url = "http://localhost:8000/api/cultural-workflow"

# Test data
test_data = {
    "persona": {
        "age": "35",
        "race": "Asian",
        "sex": "Female",
        "ancestry": "Chinese",
        "country": "United States"
    },
    "question": "How do you view democracy?",
    "options": [
        "It is the best form of government",
        "It has both advantages and disadvantages",
        "Traditional values are more important",
        "Economic development matters more than political system"
    ],
    "cultures": ["United States", "China", "India"]  # Test with 3 cultures
}

print("Testing optimized cultural workflow...")
print(f"Testing with {len(test_data['cultures'])} cultural experts")
print(f"Question: {test_data['question']}")
print("-" * 80)

start_time = time.time()
response = requests.post(url, json=test_data)
end_time = time.time()

if response.status_code == 200:
    result = response.json()
    
    print(f"\n✅ Workflow completed in {result['processing_time']:.1f} seconds")
    print(f"   (API total time: {end_time - start_time:.1f} seconds)")
    
    print(f"\n📊 Sensitivity Analysis:")
    print(f"   - Is sensitive: {result['is_sensitive']}")
    print(f"   - Sensitivity score: {result['sensitivity_score']}")
    print(f"   - Sensitive topics: {result['sensitive_topics']}")
    
    print(f"\n🌍 Cultural Routing:")
    print(f"   - Relevant cultures: {result['relevant_cultures']}")
    
    print(f"\n⏱️  Node Timing Breakdown:")
    for node, timing in result['node_times'].items():
        print(f"   - {node}: {timing:.1f}s")
    
    print(f"\n👥 Expert Responses ({len(result['expert_responses'])} experts):")
    for i, expert in enumerate(result['expert_responses'], 1):
        print(f"\n   {i}. {expert['culture']} (weight: {expert['weight']:.2f}):")
        print(f"      {expert['response'][:200]}...")
    
    print(f"\n📝 Final Response:")
    print(f"   {result['final_response']}")
    
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)