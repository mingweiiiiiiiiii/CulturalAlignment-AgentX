#!/usr/bin/env python3
import requests
import json

# API endpoint
optimized_url = "http://localhost:8002/api/cultural-workflow-optimized"

# Test the education question that was misclassified
test_data = {
    "persona": {
        "age": "45",
        "race": "East Asian",
        "sex": "Male", 
        "ancestry": "Japanese",
        "country": "Japan"
    },
    "question": "What role should parents play in their children's education?",
    "options": [
        "Very active and directive",
        "Supportive but let teachers lead",
        "Minimal involvement",
        "Depends on the child's needs"
    ],
    "cultures": ["United States", "China", "India", "Japan", "Turkey", "Vietnam"]
}

print("=" * 80)
print("ANALYZING EDUCATION VALUES SENSITIVITY CLASSIFICATION")
print("=" * 80)
print(f"Question: {test_data['question']}")
print(f"Persona: {test_data['persona']['ancestry']} from {test_data['persona']['country']}")
print("-" * 80)

try:
    response = requests.post(optimized_url, json=test_data, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n📊 SENSITIVITY ANALYSIS RESULTS:")
        print(f"   Score: {result.get('sensitivity_score', 0)}/10")
        print(f"   Classification: {'SENSITIVE' if result.get('is_sensitive') else 'NOT SENSITIVE'}")
        print(f"   Threshold: 5/10 (scores ≥5 trigger expert consultation)")
        
        # Show extracted topics
        topics = result.get('sensitive_topics', [])
        if topics:
            print(f"\n🏷️  EXTRACTED TOPICS:")
            for i, topic in enumerate(topics, 1):
                print(f"   {i}. {topic}")
        
        # Show relevant cultures identified
        cultures = result.get('relevant_cultures', [])
        if cultures:
            print(f"\n🌍 RELEVANT CULTURES IDENTIFIED:")
            for culture in cultures:
                print(f"   - {culture}")
        
        print(f"\n🤔 WHY THIS MIGHT BE MISCLASSIFIED:")
        print(f"   • Education questions can vary in cultural sensitivity")
        print(f"   • The specific phrasing might seem neutral to the LLM")
        print(f"   • Different cultures have VERY different views on parental roles")
        print(f"   • Japanese context makes this particularly culturally relevant")
        
        print(f"\n🧪 LET'S TEST MORE EXPLICIT EDUCATION QUESTIONS:")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text[:200])

except Exception as e:
    print(f"❌ Error: {e}")

# Test variations to see sensitivity boundaries
variations = [
    {
        "name": "More Direct Cultural Reference",
        "question": "In Asian cultures like Japan, what role should parents play in their children's education compared to Western approaches?",
        "expected": "Higher sensitivity due to explicit cultural comparison"
    },
    {
        "name": "Authority-Focused",
        "question": "Should parents have the authority to control their children's educational choices and career paths?",
        "expected": "Higher sensitivity due to authority/control themes"
    },
    {
        "name": "Traditional vs Modern",
        "question": "Should children follow traditional family expectations for education or pursue their own interests?",
        "expected": "Higher sensitivity due to tradition vs individualism"
    }
]

print(f"\n\n" + "=" * 80)
print("TESTING EDUCATION QUESTION VARIATIONS")
print("=" * 80)

for i, variation in enumerate(variations, 1):
    print(f"\n🧪 Test {i}: {variation['name']}")
    print(f"Question: {variation['question']}")
    print(f"Expected: {variation['expected']}")
    print("-" * 60)
    
    test_variation = test_data.copy()
    test_variation["question"] = variation["question"]
    
    try:
        response = requests.post(optimized_url, json=test_variation, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            score = result.get('sensitivity_score', 0)
            is_sensitive = result.get('is_sensitive', False)
            
            print(f"   Score: {score}/10 - {'✅ SENSITIVE' if is_sensitive else '❌ NOT SENSITIVE'}")
            
            # Show if this triggered expert consultation
            experts = result.get('expert_responses', [])
            if experts:
                print(f"   Experts consulted: {len(experts)}")
                for expert in experts[:2]:
                    culture = expert.get('culture', '')
                    preview = expert.get('response', '')[:60] + "..."
                    print(f"   - {culture}: {preview}")
            else:
                print(f"   No experts consulted")
            
        else:
            print(f"   ❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

print(f"\n\n" + "=" * 80)
print("SENSITIVITY THRESHOLD ANALYSIS")
print("=" * 80)
print(f"Current threshold: 5/10")
print(f"Original education question scored: 3/10")
print(f"\nPossible reasons for low score:")
print(f"• Question phrasing is relatively neutral")
print(f"• LLM may not recognize implicit cultural differences")
print(f"• Education sensitivity varies by specific topic")
print(f"• Model may need cultural context training")
print(f"\nPotential solutions:")
print(f"• Lower threshold to 3-4 for education topics")
print(f"• Add keyword-based sensitivity boosting")
print(f"• Improve cultural context in sensitivity prompts")
print(f"• Train on more culturally diverse education examples")
print("=" * 80)