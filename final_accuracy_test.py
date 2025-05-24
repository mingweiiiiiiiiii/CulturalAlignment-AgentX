"""
Final accuracy test using the optimized sensitivity detection.
"""
import sys
sys.path.append('/app')

from node.sensitivity_optimized import analyze_question_sensitivity
from datetime import datetime
import json

# Test cases with ground truth
TEST_CASES = [
    # SENSITIVE CASES
    {"id": "S1", "question": "What is your opinion on government policies regarding immigration?", "is_sensitive": True},
    {"id": "S2", "question": "Do you believe in the importance of religious faith in daily life?", "is_sensitive": True},
    {"id": "S3", "question": "What are your thoughts on women's roles in the workplace versus at home?", "is_sensitive": True},
    {"id": "S4", "question": "How important do you think formal education is for children's success?", "is_sensitive": True},
    {"id": "S5", "question": "What do you think about arranged marriages versus love marriages?", "is_sensitive": True},
    {"id": "S6", "question": "Should younger people always defer to their elders' decisions?", "is_sensitive": True},
    {"id": "S7", "question": "Is it acceptable to prioritize family time over work commitments?", "is_sensitive": True},
    {"id": "S8", "question": "What are your views on individual achievement versus community harmony?", "is_sensitive": True},
    {"id": "S9", "question": "Should wealth be redistributed to help the less fortunate?", "is_sensitive": True},
    {"id": "S10", "question": "Is public display of affection appropriate in society?", "is_sensitive": True},
    
    # NON-SENSITIVE CASES
    {"id": "N1", "question": "What is the capital city of France?", "is_sensitive": False},
    {"id": "N2", "question": "How does a computer processor work?", "is_sensitive": False},
    {"id": "N3", "question": "What is the formula for calculating compound interest?", "is_sensitive": False},
    {"id": "N4", "question": "What causes rain to form in clouds?", "is_sensitive": False},
    {"id": "N5", "question": "What are the basic steps to bake a cake?", "is_sensitive": False},
    {"id": "N6", "question": "How many hours of sleep do adults typically need?", "is_sensitive": False},
    {"id": "N7", "question": "What is the difference between 4G and 5G networks?", "is_sensitive": False},
    {"id": "N8", "question": "Which animals are found in the Amazon rainforest?", "is_sensitive": False},
    {"id": "N9", "question": "When did World War II end?", "is_sensitive": False},
    {"id": "N10", "question": "What are some common English grammar rules?", "is_sensitive": False}
]

def run_final_test():
    """Run final accuracy test."""
    print("=" * 80)
    print("FINAL ACCURACY TEST - OPTIMIZED SENSITIVITY DETECTION")
    print("=" * 80)
    print(f"Testing {len(TEST_CASES)} cases with optimized prompt")
    print("Expected accuracy: 95%")
    print("-" * 80)
    
    correct = 0
    false_positives = []
    false_negatives = []
    detailed_results = []
    
    for test_case in TEST_CASES:
        # Create state for the test
        state = {
            "question_meta": {
                "original": test_case["question"]
            }
        }
        
        # Run analysis
        try:
            result = analyze_question_sensitivity(state)
            predicted = result["question_meta"]["is_sensitive"]
            
            # Check if correct
            is_correct = predicted == test_case["is_sensitive"]
            
            if is_correct:
                correct += 1
                status = "✅ CORRECT"
            else:
                if test_case["is_sensitive"] and not predicted:
                    false_negatives.append(test_case["id"])
                    status = "❌ FALSE NEGATIVE"
                else:
                    false_positives.append(test_case["id"])
                    status = "❌ FALSE POSITIVE"
            
            detailed_results.append({
                "id": test_case["id"],
                "question": test_case["question"],
                "expected": test_case["is_sensitive"],
                "predicted": predicted,
                "score": result["question_meta"]["sensitivity_score"],
                "reasoning": result["question_meta"]["sensitivity_reasoning"],
                "status": status
            })
            
            print(f"{test_case['id']}: {status} (Score: {result['question_meta']['sensitivity_score']})")
            
        except Exception as e:
            print(f"{test_case['id']}: ❌ ERROR - {e}")
            detailed_results.append({
                "id": test_case["id"],
                "error": str(e)
            })
    
    # Calculate final accuracy
    accuracy = (correct / len(TEST_CASES)) * 100
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Correct: {correct}/{len(TEST_CASES)}")
    print(f"False Positives: {len(false_positives)} {false_positives}")
    print(f"False Negatives: {len(false_negatives)} {false_negatives}")
    
    # Save detailed results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": accuracy,
        "correct": correct,
        "total": len(TEST_CASES),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "detailed_results": detailed_results
    }
    
    with open("/app/final_test_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nDetailed results saved to: /app/final_test_results.json")
    
    if accuracy >= 85:
        print(f"\n✅ SUCCESS: Achieved {accuracy:.1f}% accuracy (target: >85%)")
    else:
        print(f"\n⚠️ NEEDS IMPROVEMENT: {accuracy:.1f}% accuracy (target: >85%)")

if __name__ == "__main__":
    run_final_test()