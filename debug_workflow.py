#!/usr/bin/env python3
import json
from mylanggraph.graph import create_cultural_graph
from mylanggraph.types import GraphState
import time

# Test data
persona = {
    "age": "35",
    "race": "Asian",
    "sex": "Female",
    "ancestry": "Chinese",
    "country": "United States"
}

question = "How do you view democracy?"
options = [
    "It is the best form of government",
    "It has both advantages and disadvantages",
    "Traditional values are more important",
    "Economic development matters more than political system"
]

# Format question with options
merged_question = f"{question}\n\nOptions:\n"
for i, option in enumerate(options):
    merged_question += f"{chr(65 + i)}. {option}\n"

# Create graph
graph = create_cultural_graph(cultures=["United States", "China", "India"])

# Create initial state
state: GraphState = {
    "user_profile": persona,
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

print("Running workflow with debugging...")
print(f"Initial state keys: {list(state.keys())}")
print("-" * 80)

# Run with verbose output
result = graph.invoke(state, config={
    "recursion_limit": 50,
    "configurable": {"thread_id": f"debug_{int(time.time())}"},
    "verbose": True,
})

print("\n" + "=" * 80)
print("FINAL RESULT:")
print("=" * 80)

# Check activate_set
activate_set = result.get("activate_set", [])
print(f"\n🎯 Activate set has {len(activate_set)} entries:")
for entry in activate_set:
    print(f"   - {entry}")

# Check if router was activated
print(f"\n🚦 Router activated: {result.get('activate_router', False)}")
print(f"🚦 Compose activated: {result.get('activate_compose', False)}")

# Check final response
response_state = result.get("response_state", {})
print(f"\n📝 Expert responses: {len(response_state.get('expert_responses', []))}")
print(f"📝 Final response exists: {bool(response_state.get('final', ''))}")

# Print node timing
node_times = result.get("node_times", {})
print(f"\n⏱️  Node times:")
for node, timing in node_times.items():
    print(f"   - {node}: {timing:.2f}s")