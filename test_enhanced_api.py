#!/usr/bin/env python3
import requests
import json

# Test the enhanced sensitivity system via API calls
enhanced_url = "http://localhost:8003/api/cultural-workflow-enhanced"

# Our test cases that previously failed
test_cases = [
    {
        "name": "Education Values (Was 3/10)",
        "data": {
            "persona": {
                "age": "45",
                "race": "East Asian",
                "sex": "Male",
                "ancestry": "Japanese", 
                "country": "Japan"
            },
            "question": "What role should parents play in their children's education?",
            "options": ["Very active", "Supportive", "Minimal", "Depends"],
            "cultures": ["United States", "China", "India", "Japan", "Turkey", "Vietnam"]
        },
        "original_score": 3,
        "expected_boost": "Education + Cultural context boosts should raise to 6+"
    },
    {
        "name": "Social Hierarchy (Was 4/10)", 
        "data": {
            "persona": {
                "age": "35",
                "race": "South Asian",
                "sex": "Female",
                "ancestry": "Indian",
                "country": "India"
            },
            "question": "How should younger people interact with their elders in social situations?",
            "options": ["Very formally", "Respectfully", "Casually", "Depends"],
            "cultures": ["United States", "China", "India", "Japan", "Turkey", "Vietnam"]
        },
        "original_score": 4,
        "expected_boost": "Hierarchy + Cultural context boosts should raise to 7+"
    },
    {
        "name": "Work Ethics (Was 4/10)",
        "data": {
            "persona": {
                "age": "30", 
                "race": "Asian",
                "sex": "Female",
                "ancestry": "Korean",
                "country": "South Korea"
            },
            "question": "How important is it to work long hours to show dedication?",
            "options": ["Essential", "Important", "Not important", "Varies"],
            "cultures": ["United States", "China", "India", "Japan", "Turkey", "Vietnam"]
        },
        "original_score": 4,
        "expected_boost": "Work culture + Cultural context boosts should raise to 6+"
    }
]

print("=" * 90)
print("TESTING ENHANCED SENSITIVITY WITH Q4_K_M VIA API")
print("=" * 90)
print("Comparing current system vs enhanced sensitivity detection")
print("Focus: Previously misclassified education, hierarchy, and work questions")
print()

for i, test_case in enumerate(test_cases, 1):
    print(f"\n📋 Test {i}: {test_case['name']}")
    print(f"Persona: {test_case['data']['persona']['ancestry']} from {test_case['data']['persona']['country']}")
    print(f"Question: {test_case['data']['question']}")
    print(f"Original score: {test_case['original_score']}/10")
    print(f"Expected: {test_case['expected_boost']}")
    print("-" * 80)
    
    try:
        response = requests.post(enhanced_url, json=test_case['data'], timeout=90)
        
        if response.status_code == 200:
            result = response.json()
            
            score = result.get('sensitivity_score', 0)
            is_sensitive = result.get('is_sensitive', False)
            experts_count = len(result.get('expert_responses', []))
            processing_time = result.get('processing_time', 0)
            
            # Analyze improvement
            score_improvement = score - test_case['original_score']
            
            print(f"✅ SUCCESS in {processing_time:.1f}s")
            print(f"   Score: {score}/10 (was {test_case['original_score']}/10)")
            
            if score_improvement > 0:
                print(f"   Improvement: +{score_improvement} points! 🎉")
            elif score_improvement == 0:
                print(f"   No change in score")
            else:
                print(f"   Score decreased by {abs(score_improvement)}")
            
            print(f"   Classification: {'🔴 SENSITIVE' if is_sensitive else '🟢 NOT SENSITIVE'}")
            print(f"   Threshold: ≥4 for expert consultation")
            
            if is_sensitive and experts_count > 0:
                print(f"   ✅ Experts consulted: {experts_count}")
                
                # Show expert perspectives
                for expert in result['expert_responses'][:2]:
                    culture = expert.get('culture', 'Unknown')
                    preview = expert.get('response', '')[:60] + "..."
                    print(f"      - {culture}: {preview}")
                    
            elif is_sensitive and experts_count == 0:
                print(f"   ⚠️  Marked sensitive but no experts consulted")
            else:
                print(f"   ℹ️  No expert consultation (non-sensitive)")
            
            # Show topics detected
            topics = result.get('sensitive_topics', [])
            if topics and isinstance(topics, list) and topics[0]:
                topic_preview = topics[0][:80] + "..." if len(topics[0]) > 80 else topics[0]
                print(f"   Topics: {topic_preview}")
            
            # Show final response preview
            final_response = result.get('final_response', '')
            if final_response:
                response_preview = final_response[:100] + "..."
                print(f"   Response: {response_preview}")
                
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"   Error: {response.text[:150]}")
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")

print(f"\n" + "=" * 90)
print("ENHANCED SENSITIVITY DETECTION ANALYSIS")
print("=" * 90)

print(f"\n🎯 Key Improvements Expected:")
print(f"   • Education questions: +2 boost (education topics)")
print(f"   • Hierarchy questions: +2 boost (social hierarchy topics)")
print(f"   • Work culture: +1 boost (work-related topics)")
print(f"   • Cultural context: +1 boost (Asian/cultural backgrounds)")
print(f"   • Lower threshold: 4/10 instead of 5/10")

print(f"\n📊 Success Criteria:")
print(f"   • Education Values: 3→6+ (should trigger experts)")
print(f"   • Social Hierarchy: 4→7+ (should trigger experts)")
print(f"   • Work Ethics: 4→6+ (should trigger experts)")

print(f"\n✨ Q4_K_M Performance Notes:")
print(f"   • Enhanced prompts test model's cultural reasoning")
print(f"   • Complex context handling with quantized model")
print(f"   • Maintained response quality with optimization")

print("=" * 90)