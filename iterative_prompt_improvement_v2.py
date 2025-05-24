"""
Iterative prompt improvement for cultural sensitivity detection.
Uses local ollama client with granite3.3 model.
"""
import sys
sys.path.append('/app')

from typing import Dict, List, Tuple
from node.combined_analysis_fixed import analyze_question_sensitivity
from llmagentsetting.ollama_client import OllamaClient
import json
import re
from datetime import datetime

# Test cases with ground truth
TEST_CASES = [
    # SENSITIVE CASES (10 total)
    {
        "id": "S1",
        "category": "Political/Governance",
        "question": "What is your opinion on government policies regarding immigration?",
        "is_sensitive": True,
        "reason": "Immigration policies vary greatly between cultures and are politically charged"
    },
    {
        "id": "S2", 
        "category": "Religious/Values",
        "question": "Do you believe in the importance of religious faith in daily life?",
        "is_sensitive": True,
        "reason": "Religious beliefs and their role in life vary significantly across cultures"
    },
    {
        "id": "S3",
        "category": "Gender Roles",
        "question": "What are your thoughts on women's roles in the workplace versus at home?",
        "is_sensitive": True,
        "reason": "Gender roles and expectations differ dramatically between cultures"
    },
    {
        "id": "S4",
        "category": "Education/Values",
        "question": "How important do you think formal education is for children's success?",
        "is_sensitive": True,
        "reason": "Educational values and importance vary by culture and socioeconomic context"
    },
    {
        "id": "S5",
        "category": "Family Structure",
        "question": "What do you think about arranged marriages versus love marriages?",
        "is_sensitive": True,
        "reason": "Marriage customs are deeply cultural with no universal norm"
    },
    {
        "id": "S6",
        "category": "Authority/Hierarchy",
        "question": "Should younger people always defer to their elders' decisions?",
        "is_sensitive": True,
        "reason": "Attitudes toward age and authority vary significantly between cultures"
    },
    {
        "id": "S7",
        "category": "Work Culture",
        "question": "Is it acceptable to prioritize family time over work commitments?",
        "is_sensitive": True,
        "reason": "Work-life balance expectations differ greatly across cultures"
    },
    {
        "id": "S8",
        "category": "Social Values",
        "question": "What are your views on individual achievement versus community harmony?",
        "is_sensitive": True,
        "reason": "Individualism vs collectivism is a fundamental cultural difference"
    },
    {
        "id": "S9",
        "category": "Economic Values",
        "question": "Should wealth be redistributed to help the less fortunate?",
        "is_sensitive": True,
        "reason": "Economic philosophies and welfare attitudes vary by culture"
    },
    {
        "id": "S10",
        "category": "Social Norms",
        "question": "Is public display of affection appropriate in society?",
        "is_sensitive": True,
        "reason": "Attitudes toward public affection vary dramatically by culture"
    },
    
    # NON-SENSITIVE CASES (10 total)
    {
        "id": "N1",
        "category": "Factual",
        "question": "What is the capital city of France?",
        "is_sensitive": False,
        "reason": "Objective geographical fact"
    },
    {
        "id": "N2",
        "category": "Technical",
        "question": "How does a computer processor work?",
        "is_sensitive": False,
        "reason": "Technical explanation with no cultural implications"
    },
    {
        "id": "N3",
        "category": "Mathematical",
        "question": "What is the formula for calculating compound interest?",
        "is_sensitive": False,
        "reason": "Mathematical concept universal across cultures"
    },
    {
        "id": "N4",
        "category": "Scientific",
        "question": "What causes rain to form in clouds?",
        "is_sensitive": False,
        "reason": "Natural phenomenon with scientific explanation"
    },
    {
        "id": "N5",
        "category": "Practical",
        "question": "What are the basic steps to bake a cake?",
        "is_sensitive": False,
        "reason": "General cooking instruction without cultural specifics"
    },
    {
        "id": "N6",
        "category": "Health/Universal",
        "question": "How many hours of sleep do adults typically need?",
        "is_sensitive": False,
        "reason": "Basic health information applicable universally"
    },
    {
        "id": "N7",
        "category": "Technology",
        "question": "What is the difference between 4G and 5G networks?",
        "is_sensitive": False,
        "reason": "Technical comparison without cultural context"
    },
    {
        "id": "N8",
        "category": "Nature",
        "question": "Which animals are found in the Amazon rainforest?",
        "is_sensitive": False,
        "reason": "Factual information about wildlife"
    },
    {
        "id": "N9",
        "category": "Historical Fact",
        "question": "When did World War II end?",
        "is_sensitive": False,
        "reason": "Historical date that is factually established"
    },
    {
        "id": "N10",
        "category": "Language/Neutral",
        "question": "What are some common English grammar rules?",
        "is_sensitive": False,
        "reason": "Language mechanics without cultural judgment"
    }
]

# Different prompt strategies
PROMPT_STRATEGIES = {
    "baseline": {
        "name": "Baseline - Simple JSON Request",
        "prompt": """Analyze the following question for cultural sensitivity.

Question: {question}

Respond with JSON:
{{
    "sensitivity_score": <0-10>,
    "is_sensitive": <true/false>,
    "sensitive_topics": [],
    "relevant_cultures": [],
    "reasoning": ""
}}"""
    },
    
    "structured": {
        "name": "Structured - Detailed Guidelines",
        "prompt": """Analyze this question for cultural sensitivity by checking if it touches on topics where different cultures may have conflicting values or beliefs.

Question: {question}

Check for these cultural dimensions:
1. Religious beliefs and practices
2. Political systems and governance
3. Gender roles and relationships
4. Family structures and values
5. Work-life balance and priorities
6. Social hierarchies and authority
7. Individual vs collective values
8. Economic systems and wealth

Return JSON with this exact format:
{{
    "sensitivity_score": <integer 0-10>,
    "is_sensitive": <boolean>,
    "sensitive_topics": <array of strings>,
    "relevant_cultures": <array of strings>,
    "reasoning": <string explanation>
}}

Score 5+ means culturally sensitive."""
    },
    
    "examples": {
        "name": "Examples - Few-shot Learning",
        "prompt": """Analyze cultural sensitivity of questions. Here are examples:

SENSITIVE (score 7): "Should women work or stay home?" - Gender roles vary by culture
NOT SENSITIVE (score 2): "What is the capital of Japan?" - Factual question

SENSITIVE (score 8): "Is democracy the best system?" - Political values differ
NOT SENSITIVE (score 1): "How do computers work?" - Technical topic

Question: {question}

Return JSON:
{{
    "sensitivity_score": <0-10>,
    "is_sensitive": <true if score>=5>,
    "sensitive_topics": [],
    "relevant_cultures": [],
    "reasoning": ""
}}"""
    },
    
    "direct": {
        "name": "Direct - Explicit Instructions",
        "prompt": """You are analyzing if a question might receive different answers from people of different cultural backgrounds.

Question: {question}

Questions are culturally sensitive if they:
- Involve values, beliefs, or practices that vary between cultures
- Touch on politics, religion, gender, family, or social norms
- Could offend or exclude certain cultural groups

Questions are NOT sensitive if they:
- Ask for facts, technical information, or universal concepts
- Have answers that don't depend on cultural background

Analyze and return ONLY valid JSON:
{{
    "sensitivity_score": 0-10 integer,
    "is_sensitive": true if score >= 5,
    "sensitive_topics": ["topic1", "topic2"],
    "relevant_cultures": ["culture1", "culture2"],
    "reasoning": "brief explanation"
}}"""
    },
    
    "focused": {
        "name": "Focused - Key Indicators",
        "prompt": """Determine if this question is culturally sensitive.

Question: {question}

A question is culturally sensitive if it asks about:
- Personal values or beliefs
- Social norms or customs  
- Political or religious views
- Gender or family roles
- Work-life priorities
- Authority and hierarchy

Return this JSON structure:
{{
    "sensitivity_score": <0-10 where 5+ is sensitive>,
    "is_sensitive": <true/false based on score>,
    "sensitive_topics": <list of topics>,
    "relevant_cultures": <list of cultures>,
    "reasoning": <one sentence explanation>
}}

Respond with JSON only."""
    }
}

def test_single_prompt(prompt_template: str, test_case: Dict, client: OllamaClient) -> Tuple[bool, Dict]:
    """Test a single prompt on a single test case."""
    try:
        # Format prompt with the question
        prompt = prompt_template.format(question=test_case["question"])
        
        # Get response from LLM
        response = client.generate(prompt)
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            predicted_sensitive = result.get("is_sensitive", False)
            
            # Check if prediction matches ground truth
            correct = predicted_sensitive == test_case["is_sensitive"]
            
            return correct, result
        else:
            print(f"Failed to parse JSON from response: {response[:100]}...")
            return False, {"error": "JSON parsing failed"}
            
    except Exception as e:
        print(f"Error testing prompt: {e}")
        return False, {"error": str(e)}

def evaluate_prompt_strategy(strategy_name: str, strategy: Dict, test_cases: List[Dict], client: OllamaClient, sample_size: int = 5) -> Dict:
    """Evaluate a prompt strategy on a sample of test cases."""
    prompt_template = strategy["prompt"]
    
    # Test on a subset first
    sample_cases = test_cases[:sample_size]
    correct = 0
    false_positives = 0
    false_negatives = 0
    
    print(f"\n   Testing on {sample_size} sample cases...")
    
    for test_case in sample_cases:
        is_correct, result = test_single_prompt(prompt_template, test_case, client)
        
        if is_correct:
            correct += 1
        else:
            if test_case["is_sensitive"] and not result.get("is_sensitive", False):
                false_negatives += 1
            elif not test_case["is_sensitive"] and result.get("is_sensitive", False):
                false_positives += 1
    
    accuracy = (correct / sample_size) * 100
    
    return {
        "strategy": strategy_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": sample_size,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def run_full_evaluation(strategy_name: str, strategy: Dict, test_cases: List[Dict], client: OllamaClient) -> Dict:
    """Run full evaluation on all test cases."""
    prompt_template = strategy["prompt"]
    
    correct = 0
    false_positives = 0
    false_negatives = 0
    errors = 0
    
    print(f"\n   Running full evaluation on {len(test_cases)} cases...")
    
    for i, test_case in enumerate(test_cases):
        is_correct, result = test_single_prompt(prompt_template, test_case, client)
        
        if "error" in result:
            errors += 1
        elif is_correct:
            correct += 1
        else:
            if test_case["is_sensitive"] and not result.get("is_sensitive", False):
                false_negatives += 1
                print(f"     FN: {test_case['id']} - {test_case['category']}")
            elif not test_case["is_sensitive"] and result.get("is_sensitive", False):
                false_positives += 1
                print(f"     FP: {test_case['id']} - {test_case['category']}")
    
    accuracy = (correct / len(test_cases)) * 100
    
    return {
        "strategy": strategy_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "errors": errors
    }

def main():
    """Run iterative prompt improvement."""
    print("=" * 80)
    print("ITERATIVE PROMPT IMPROVEMENT FOR CULTURAL SENSITIVITY DETECTION")
    print("=" * 80)
    print("Target: >85% accuracy")
    print("Model: granite3.3:latest (local ollama)")
    print("-" * 80)
    
    # Initialize client
    client = OllamaClient()
    
    best_accuracy = 0
    best_strategy = None
    best_results = None
    
    # Test each strategy
    for i, (strategy_name, strategy) in enumerate(PROMPT_STRATEGIES.items(), 1):
        print(f"\n🔄 Iteration {i}: Testing '{strategy['name']}'")
        
        # First test on a small sample
        sample_results = evaluate_prompt_strategy(
            strategy_name, strategy, TEST_CASES, client, sample_size=6
        )
        
        print(f"   Sample accuracy: {sample_results['accuracy']:.1f}%")
        print(f"   FP: {sample_results['false_positives']}, FN: {sample_results['false_negatives']}")
        
        # If sample looks promising (>60%), run full test
        if sample_results['accuracy'] >= 60:
            print(f"   ✅ Promising! Running full evaluation...")
            
            full_results = run_full_evaluation(
                strategy_name, strategy, TEST_CASES, client
            )
            
            print(f"\n   Full accuracy: {full_results['accuracy']:.1f}%")
            print(f"   Correct: {full_results['correct']}/{full_results['total']}")
            print(f"   False Positives: {full_results['false_positives']}")
            print(f"   False Negatives: {full_results['false_negatives']}")
            print(f"   Errors: {full_results['errors']}")
            
            if full_results['accuracy'] > best_accuracy:
                best_accuracy = full_results['accuracy']
                best_strategy = strategy_name
                best_results = full_results
                print(f"   🏆 New best! {best_accuracy:.1f}%")
                
                # Check if we've reached target
                if best_accuracy >= 85:
                    print(f"\n✅ TARGET ACHIEVED! {best_accuracy:.1f}% accuracy")
                    break
        else:
            print(f"   ❌ Not promising enough, skipping full evaluation")
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if best_strategy:
        print(f"Best Strategy: {PROMPT_STRATEGIES[best_strategy]['name']}")
        print(f"Best Accuracy: {best_accuracy:.1f}%")
        print(f"Details: {best_results}")
        
        # Save best prompt
        with open("/app/best_prompt_result.json", "w") as f:
            json.dump({
                "strategy": best_strategy,
                "prompt": PROMPT_STRATEGIES[best_strategy]["prompt"],
                "accuracy": best_accuracy,
                "results": best_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\nBest prompt saved to: /app/best_prompt_result.json")
        
        if best_accuracy < 85:
            print(f"\n⚠️ Target not achieved. Gap: {85 - best_accuracy:.1f}%")
            print("Consider adding more prompt strategies or fine-tuning the model.")
    else:
        print("No successful strategies found!")

if __name__ == "__main__":
    main()