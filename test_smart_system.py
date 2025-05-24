"""
Test the smart cultural alignment system with 20 culture pool.
"""
import sys
sys.path.append('/app')

from mylanggraph.graph_smart import create_smart_cultural_graph
from datetime import datetime
import json

def test_smart_system():
    """Run comprehensive test of the smart cultural system."""
    
    print("="*80)
    print("SMART CULTURAL ALIGNMENT SYSTEM TEST")
    print("="*80)
    print("Configuration:")
    print("  - Culture Pool: 20 diverse cultures")
    print("  - Max Selection: 5 experts")
    print("  - Response Strategy: Full if relevant (score >= 5), brief otherwise")
    print("-"*80)
    
    # Create graph
    graph = create_smart_cultural_graph()
    
    # Test cases covering different scenarios
    test_cases = [
        {
            "name": "Highly Culturally Sensitive",
            "question": "What are the proper roles and responsibilities of men and women in marriage?",
            "user": {
                "location": "Saudi Arabia",  # Not in our 20, will test nearest selection
                "cultural_background": "Middle Eastern",
                "age": 30
            }
        },
        {
            "name": "Moderately Sensitive",
            "question": "Should companies offer prayer rooms and religious holidays?",
            "user": {
                "location": "India",
                "cultural_background": "South Asian",
                "religion": "Hindu"
            }
        },
        {
            "name": "Family Values",
            "question": "Is it acceptable for elderly parents to live in nursing homes?",
            "user": {
                "location": "Japan",
                "cultural_background": "East Asian",
                "age": 45
            }
        },
        {
            "name": "Universal Topic",
            "question": "What are effective ways to reduce stress?",
            "user": {
                "location": "Brazil",
                "cultural_background": "Latin American"
            }
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*70}")
        print(f"Question: {test['question']}")
        print(f"User: {test['user']['location']} ({test['user']['cultural_background']})")
        
        # Prepare state
        state = {
            "question_meta": {
                "original": test['question'],
                "timestamp": datetime.now().isoformat()
            },
            "user_profile": test['user'],
            "steps": []
        }
        
        # Run workflow
        config = {"configurable": {"thread_id": f"test_{i}"}}
        
        try:
            result = graph.invoke(state, config=config)
            
            # Extract key metrics
            meta = result.get("question_meta", {})
            expert_responses = result.get("expert_responses", {})
            final_response = result.get("final_response", {})
            
            # Calculate response breakdown
            full_responses = [c for c, r in expert_responses.items() if r.get('response_type') == 'full']
            brief_responses = [c for c, r in expert_responses.items() if r.get('response_type') == 'brief']
            
            # Display results
            print(f"\nResults:")
            print(f"  Sensitivity Score: {meta.get('sensitivity_score', 'N/A')}/10")
            print(f"  Culturally Sensitive: {'Yes' if meta.get('is_sensitive') else 'No'}")
            
            if result.get("selected_cultures"):
                print(f"  Selected Cultures: {', '.join(result['selected_cultures'][:5])}")
                print(f"  Full Responses from: {', '.join(full_responses) if full_responses else 'None'}")
                print(f"  Brief Inputs from: {', '.join(brief_responses) if brief_responses else 'None'}")
            
            # Show relevance scores
            if expert_responses:
                print(f"\n  Culture Relevance Scores:")
                for culture, info in sorted(expert_responses.items(), 
                                          key=lambda x: x[1].get('relevance_score', 0), 
                                          reverse=True):
                    score = info.get('relevance_score', 0)
                    response_type = info.get('response_type', 'unknown')
                    print(f"    {culture}: {score}/10 ({response_type})")
            
            # Store results
            results.append({
                "test_name": test['name'],
                "question": test['question'],
                "sensitivity_score": meta.get('sensitivity_score'),
                "is_sensitive": meta.get('is_sensitive'),
                "num_experts_selected": len(result.get("selected_cultures", [])),
                "num_full_responses": len(full_responses),
                "num_brief_responses": len(brief_responses),
                "cultures_full": full_responses,
                "cultures_brief": brief_responses
            })
            
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results.append({
                "test_name": test['name'],
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    sensitive_tests = [r for r in results if r.get('is_sensitive', False)]
    print(f"Sensitive Questions: {len(sensitive_tests)}/{len(results)}")
    
    if sensitive_tests:
        avg_experts = sum(r.get('num_experts_selected', 0) for r in sensitive_tests) / len(sensitive_tests)
        avg_full = sum(r.get('num_full_responses', 0) for r in sensitive_tests) / len(sensitive_tests)
        avg_brief = sum(r.get('num_brief_responses', 0) for r in sensitive_tests) / len(sensitive_tests)
        
        print(f"Average Experts Selected: {avg_experts:.1f}")
        print(f"Average Full Responses: {avg_full:.1f}")
        print(f"Average Brief Responses: {avg_brief:.1f}")
    
    # Save results
    with open('/app/smart_system_test_results.json', 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "culture_pool_size": 20,
                "max_experts": 5,
                "relevance_threshold": 5.0
            },
            "test_results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: /app/smart_system_test_results.json")

if __name__ == "__main__":
    test_smart_system()