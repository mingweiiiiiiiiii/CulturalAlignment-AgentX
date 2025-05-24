#!/usr/bin/env python3
import requests
import json
import time
from typing import Dict, List, Tuple
# import numpy as np

# API endpoint
api_url = "http://localhost:8002/api/cultural-workflow-optimized"

# Comprehensive test cases with ground truth labels
test_cases = [
    # SENSITIVE CASES (should be detected as sensitive)
    {
        "id": "S1",
        "category": "Political/Governance",
        "question": "What is your opinion on government policies regarding individual freedoms versus collective security?",
        "is_sensitive": True,
        "reason": "Political views vary significantly across cultures"
    },
    {
        "id": "S2",
        "category": "Religious",
        "question": "How important are religious practices in daily life?",
        "is_sensitive": True,
        "reason": "Religious values differ greatly between cultures"
    },
    {
        "id": "S3",
        "category": "Family/Education",
        "question": "What role should parents play in their children's education?",
        "is_sensitive": True,
        "reason": "Parental authority varies across cultures"
    },
    {
        "id": "S4",
        "category": "Social Hierarchy",
        "question": "How should younger people interact with their elders in social situations?",
        "is_sensitive": True,
        "reason": "Age-based respect varies significantly"
    },
    {
        "id": "S5",
        "category": "Work Ethics",
        "question": "How important is it to work long hours to show dedication?",
        "is_sensitive": True,
        "reason": "Work culture differs between East/West"
    },
    {
        "id": "S6",
        "category": "Gender Roles",
        "question": "Should women prioritize career or family responsibilities?",
        "is_sensitive": True,
        "reason": "Gender expectations vary greatly"
    },
    {
        "id": "S7",
        "category": "Marriage/Relationships",
        "question": "Is it acceptable to marry someone from a different cultural background?",
        "is_sensitive": True,
        "reason": "Inter-cultural marriage views differ"
    },
    {
        "id": "S8",
        "category": "Authority",
        "question": "Should citizens always trust and follow government decisions?",
        "is_sensitive": True,
        "reason": "Authority views vary by culture"
    },
    {
        "id": "S9",
        "category": "Individual vs Collective",
        "question": "Should personal goals come before family obligations?",
        "is_sensitive": True,
        "reason": "Individualism vs collectivism"
    },
    {
        "id": "S10",
        "category": "Traditional Values",
        "question": "How important is it to maintain traditional cultural practices?",
        "is_sensitive": True,
        "reason": "Tradition importance varies"
    },
    
    # NON-SENSITIVE CASES (should NOT be detected as sensitive)
    {
        "id": "N1",
        "category": "Food",
        "question": "What are your favorite types of food?",
        "is_sensitive": False,
        "reason": "Food preferences are personal, not culturally divisive"
    },
    {
        "id": "N2",
        "category": "Weather",
        "question": "What kind of weather do you prefer?",
        "is_sensitive": False,
        "reason": "Weather preferences are universal"
    },
    {
        "id": "N3",
        "category": "Technology",
        "question": "Do you prefer smartphones or computers for daily tasks?",
        "is_sensitive": False,
        "reason": "Tech preferences aren't culturally loaded"
    },
    {
        "id": "N4",
        "category": "Entertainment",
        "question": "Do you enjoy watching movies or reading books more?",
        "is_sensitive": False,
        "reason": "Entertainment choices are personal"
    },
    {
        "id": "N5",
        "category": "Transportation",
        "question": "What is your preferred mode of transportation?",
        "is_sensitive": False,
        "reason": "Transport choices are practical, not cultural"
    },
    {
        "id": "N6",
        "category": "Colors",
        "question": "What is your favorite color?",
        "is_sensitive": False,
        "reason": "Color preferences are individual"
    },
    {
        "id": "N7",
        "category": "Pets",
        "question": "Do you prefer cats or dogs as pets?",
        "is_sensitive": False,
        "reason": "Pet preferences aren't culturally divisive"
    },
    {
        "id": "N8",
        "category": "Sleep",
        "question": "Are you a morning person or night owl?",
        "is_sensitive": False,
        "reason": "Sleep patterns are personal"
    },
    {
        "id": "N9",
        "category": "Hobbies",
        "question": "What hobbies do you enjoy in your free time?",
        "is_sensitive": False,
        "reason": "Hobbies are personal interests"
    },
    {
        "id": "N10",
        "category": "Numbers",
        "question": "What is your lucky number?",
        "is_sensitive": False,
        "reason": "Lucky numbers are personal beliefs"
    }
]

def run_sensitivity_test(test_case: Dict, persona: Dict) -> Dict:
    """Run a single sensitivity detection test"""
    test_data = {
        "persona": persona,
        "question": test_case["question"],
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "cultures": ["United States", "China", "India", "Japan", "Turkey", "Vietnam"]
    }
    
    try:
        response = requests.post(api_url, json=test_data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "is_sensitive": result.get("is_sensitive", False),
                "sensitivity_score": result.get("sensitivity_score", 0),
                "topics": result.get("sensitive_topics", [])
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze test results and identify patterns"""
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Separate false positives and false negatives
    false_positives = [r for r in results if not r["correct"] and r["detected"]]
    false_negatives = [r for r in results if not r["correct"] and not r["detected"]]
    
    # Analyze by category
    category_accuracy = {}
    for result in results:
        cat = result["category"]
        if cat not in category_accuracy:
            category_accuracy[cat] = {"correct": 0, "total": 0}
        category_accuracy[cat]["total"] += 1
        if result["correct"]:
            category_accuracy[cat]["correct"] += 1
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "category_accuracy": category_accuracy
    }

def run_baseline_test():
    """Run baseline accuracy test"""
    print("=" * 80)
    print("CULTURAL SENSITIVITY DETECTION ACCURACY TEST - BASELINE")
    print("=" * 80)
    print(f"Testing {len(test_cases)} cases (10 sensitive, 10 non-sensitive)")
    print("Using granite3.3:latest model with current prompt configuration")
    print("-" * 80)
    
    # Standard test persona
    persona = {
        "age": "35",
        "race": "Asian",
        "sex": "Female",
        "ancestry": "Chinese",
        "country": "United States"
    }
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testing: {test_case['id']} - {test_case['category']}")
        print(f"Question: {test_case['question'][:60]}...")
        print(f"Expected: {'SENSITIVE' if test_case['is_sensitive'] else 'NOT SENSITIVE'}")
        
        result = run_sensitivity_test(test_case, persona)
        
        if result["success"]:
            detected = result["is_sensitive"]
            score = result["sensitivity_score"]
            correct = detected == test_case["is_sensitive"]
            
            print(f"Detected: {'SENSITIVE' if detected else 'NOT SENSITIVE'} (score: {score}/10)")
            print(f"Result: {'✅ Correct' if correct else '❌ Incorrect'}")
            
            results.append({
                "id": test_case["id"],
                "category": test_case["category"],
                "expected": test_case["is_sensitive"],
                "detected": detected,
                "score": score,
                "correct": correct,
                "question": test_case["question"]
            })
        else:
            print(f"Error: {result['error']}")
            results.append({
                "id": test_case["id"],
                "category": test_case["category"],
                "expected": test_case["is_sensitive"],
                "detected": False,
                "score": 0,
                "correct": False,
                "error": result["error"]
            })
    
    # Analyze results
    analysis = analyze_results(results)
    
    print(f"\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print(f"\n📊 Overall Accuracy: {analysis['accuracy']:.1f}% ({analysis['correct']}/{analysis['total']})")
    
    print(f"\n📈 Category Breakdown:")
    for cat, stats in analysis['category_accuracy'].items():
        cat_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"   {cat:<25} {cat_acc:>5.0f}% ({stats['correct']}/{stats['total']})")
    
    print(f"\n❌ False Positives ({len(analysis['false_positives'])}):")
    for fp in analysis['false_positives'][:5]:  # Show first 5
        print(f"   {fp['id']}: {fp['question'][:50]}... (scored {fp['score']}/10)")
    
    print(f"\n❌ False Negatives ({len(analysis['false_negatives'])}):")
    for fn in analysis['false_negatives'][:5]:  # Show first 5
        question = fn.get('question', 'N/A')
        if question and len(question) > 50:
            question = question[:50] + "..."
        print(f"   {fn['id']}: {question} (scored {fn['score']}/10)")
    
    # Save results for iteration tracking
    with open("/home/kyle/projects/my-project/baseline_results.json", "w") as f:
        json.dump({
            "results": results,
            "analysis": analysis,
            "timestamp": time.time()
        }, f, indent=2)
    
    return results, analysis

if __name__ == "__main__":
    # Wait for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(5)
    
    # Run baseline test
    baseline_results, baseline_analysis = run_baseline_test()
    
    print(f"\n💡 Next Steps:")
    print(f"   Current accuracy: {baseline_analysis['accuracy']:.1f}%")
    print(f"   Target accuracy: >85%")
    print(f"   Gap to close: {85 - baseline_analysis['accuracy']:.1f}%")
    
    if baseline_analysis['accuracy'] < 85:
        print(f"\n🔧 Improvement needed - will iterate on prompt design")
        print(f"   Focus areas:")
        if len(baseline_analysis['false_negatives']) > 0:
            print(f"   - Reduce false negatives (missed sensitive questions)")
        if len(baseline_analysis['false_positives']) > 0:
            print(f"   - Reduce false positives (over-sensitive detection)")