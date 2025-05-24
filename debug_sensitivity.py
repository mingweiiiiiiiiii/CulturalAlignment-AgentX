#!/usr/bin/env python3
from node.sen_agent_node import determine_cultural_sensitivity
from mylanggraph.types import GraphState

# Test data
state: GraphState = {
    "user_profile": {
        "age": "35",
        "race": "Asian",
        "sex": "Female",
        "ancestry": "Chinese",
        "country": "United States"
    },
    "question_meta": {
        "original": """How do you view democracy?

Options:
A. It is the best form of government
B. It has both advantages and disadvantages
C. Traditional values are more important
D. Economic development matters more than political system
""",
        "options": [
            "It is the best form of government",
            "It has both advantages and disadvantages",
            "Traditional values are more important",
            "Economic development matters more than political system"
        ],
        "sensitive_topics": [],
        "relevant_cultures": [],
    }
}

print("Testing sensitivity check...")
print(f"Question: {state['question_meta']['original']}")
print("-" * 80)

result = determine_cultural_sensitivity(state)

print(f"\nSensitivity results:")
print(f"Is sensitive: {result['question_meta'].get('is_sensitive', 'Not set')}")
print(f"Sensitivity score: {result['question_meta'].get('sensitivity_score', 'Not set')}")
print(f"Is sensitive (direct): {result.get('is_sensitive', 'Not set')}")

# Try with a more obviously sensitive question
state2 = state.copy()
state2["question_meta"]["original"] = """What is your opinion on China's government policies?

Options:
A. They are excellent
B. They need improvement
C. I prefer not to say
D. Democracy is better
"""

print("\n" + "=" * 80)
print("Testing with more sensitive question:")
print(f"Question: {state2['question_meta']['original']}")
print("-" * 80)

result2 = determine_cultural_sensitivity(state2)
print(f"\nSensitivity results:")
print(f"Is sensitive: {result2['question_meta'].get('is_sensitive', 'Not set')}")
print(f"Sensitivity score: {result2['question_meta'].get('sensitivity_score', 'Not set')}")
print(f"Is sensitive (direct): {result2.get('is_sensitive', 'Not set')}")