"""
Comprehensive benchmark comparing:
1. Cultural alignment system (20 culture pool)
2. Direct LLM response (no cultural workflow)
3. Baseline essay generator

Shows full responses from each approach.
"""
import sys
sys.path.append('/app')

import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

from mylanggraph.graph_smart import create_smart_cultural_graph
from llmagentsetting.ollama_client import OllamaClient
from utility.inputData import PersonaSampler

def get_direct_llm_response(question: str, user_profile: Dict) -> Tuple[str, float]:
    """Get direct response from LLM without cultural workflow."""
    client = OllamaClient()
    
    prompt = f"""Answer the following question thoughtfully:

Question: {question}

User Context:
- Location: {user_profile.get('location', 'Unknown')}
- Age: {user_profile.get('age', 'Unknown')}
- Cultural Background: {user_profile.get('cultural_background', 'Unknown')}

Please provide a balanced and comprehensive response:"""
    
    start_time = time.time()
    try:
        response = client.generate(prompt)
        latency = time.time() - start_time
        return response, latency
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def get_baseline_response(question: str, user_profile: Dict) -> Tuple[str, float]:
    """Get baseline essay response."""
    client = OllamaClient()
    
    prompt = f"""Write a balanced essay answering this question:

{question}

Consider multiple perspectives and provide a thoughtful analysis:"""
    
    start_time = time.time()
    try:
        response = client.generate(prompt)
        latency = time.time() - start_time
        return response, latency
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def get_cultural_alignment_response(question: str, user_profile: Dict, graph) -> Tuple[Dict, float]:
    """Get response from cultural alignment system."""
    state = {
        "question_meta": {
            "original": question,
            "timestamp": datetime.now().isoformat()
        },
        "user_profile": user_profile,
        "steps": []
    }
    
    start_time = time.time()
    try:
        config = {"configurable": {"thread_id": f"benchmark_{int(time.time())}"}}
        result = graph.invoke(state, config=config)
        latency = time.time() - start_time
        return result, latency
    except Exception as e:
        return {"error": str(e)}, 0.0

def format_cultural_response(result: Dict) -> Dict:
    """Extract key information from cultural alignment response."""
    meta = result.get("question_meta", {})
    final = result.get("final_response", {})
    expert_responses = result.get("expert_responses", {})
    
    # Count full vs brief responses
    full_experts = [c for c, r in expert_responses.items() if r.get('response_type') == 'full']
    brief_experts = [c for c, r in expert_responses.items() if r.get('response_type') == 'brief']
    
    return {
        "sensitivity_score": meta.get("sensitivity_score", "N/A"),
        "is_sensitive": meta.get("is_sensitive", False),
        "num_experts": len(expert_responses),
        "full_responses": full_experts,
        "brief_responses": brief_experts,
        "main_response": final.get("main_response", "No response generated"),
        "cultural_insights": final.get("cultural_insights", [])
    }

def run_comparison_benchmark(n_tests: int = 5):
    """Run comprehensive comparison benchmark."""
    print("="*80)
    print("CULTURAL ALIGNMENT SYSTEM - COMPREHENSIVE COMPARISON")
    print("="*80)
    print("Comparing 3 approaches:")
    print("1. Cultural Alignment (20 culture pool, smart selection)")
    print("2. Direct LLM (no cultural workflow)")
    print("3. Baseline Essay Generator")
    print("-"*80)
    
    # Initialize
    sampler = PersonaSampler()
    graph = create_smart_cultural_graph()
    results = []
    
    # Sample profiles and questions
    profiles = sampler.sample_profiles(n_tests)
    
    for i in range(n_tests):
        question, options = sampler.sample_question()
        merged_question = f"{question}\n\nOptions:\n" + "\n".join([
            f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)
        ])
        
        print(f"\n{'='*70}")
        print(f"TEST {i+1}/{n_tests}")
        print(f"{'='*70}")
        print(f"\nQuestion: {question}")
        print(f"\nUser Profile:")
        print(f"  Location: {profiles[i].get('place of birth', 'Unknown')}")
        print(f"  Age: {profiles[i].get('age', 'Unknown')}")
        print(f"  Background: {profiles[i].get('race', 'Unknown')}")
        print("-"*70)
        
        test_result = {
            "test_id": i + 1,
            "question": question,
            "user_profile": {
                "location": profiles[i].get('place of birth', 'Unknown'),
                "age": profiles[i].get('age', 'Unknown'),
                "background": profiles[i].get('race', 'Unknown')
            }
        }
        
        # 1. CULTURAL ALIGNMENT SYSTEM
        print("\n1. CULTURAL ALIGNMENT SYSTEM")
        print("-"*40)
        cultural_result, cultural_latency = get_cultural_alignment_response(
            merged_question, profiles[i], graph
        )
        
        if "error" not in cultural_result:
            formatted = format_cultural_response(cultural_result)
            print(f"Sensitivity: {formatted['is_sensitive']} (Score: {formatted['sensitivity_score']}/10)")
            print(f"Experts Consulted: {formatted['num_experts']}")
            if formatted['full_responses']:
                print(f"Full Responses from: {', '.join(formatted['full_responses'])}")
            if formatted['brief_responses']:
                print(f"Brief Inputs from: {', '.join(formatted['brief_responses'])}")
            print(f"\nResponse Preview:")
            print(formatted['main_response'][:300] + "...")
            
            test_result['cultural_alignment'] = {
                "latency": cultural_latency,
                "sensitivity_score": formatted['sensitivity_score'],
                "is_sensitive": formatted['is_sensitive'],
                "num_experts": formatted['num_experts'],
                "full_responses": formatted['full_responses'],
                "brief_responses": formatted['brief_responses'],
                "response": formatted['main_response'],
                "response_length": len(formatted['main_response']),
                "cultural_insights": formatted['cultural_insights']
            }
        else:
            print(f"Error: {cultural_result['error']}")
            test_result['cultural_alignment'] = {"error": cultural_result['error']}
        
        # 2. DIRECT LLM RESPONSE
        print("\n\n2. DIRECT LLM RESPONSE")
        print("-"*40)
        direct_response, direct_latency = get_direct_llm_response(merged_question, profiles[i])
        print(f"Response Preview:")
        print(direct_response[:300] + "...")
        
        test_result['direct_llm'] = {
            "latency": direct_latency,
            "response": direct_response,
            "response_length": len(direct_response)
        }
        
        # 3. BASELINE ESSAY
        print("\n\n3. BASELINE ESSAY")
        print("-"*40)
        baseline_response, baseline_latency = get_baseline_response(merged_question, profiles[i])
        print(f"Response Preview:")
        print(baseline_response[:300] + "...")
        
        test_result['baseline'] = {
            "latency": baseline_latency,
            "response": baseline_response,
            "response_length": len(baseline_response)
        }
        
        # Store results
        results.append(test_result)
        
        # Performance comparison
        print(f"\n\nPERFORMANCE SUMMARY:")
        print(f"  Cultural Alignment: {cultural_latency:.1f}s")
        print(f"  Direct LLM: {direct_latency:.1f}s")
        print(f"  Baseline: {baseline_latency:.1f}s")
    
    return results

def generate_comparison_report(results: List[Dict]) -> str:
    """Generate detailed comparison report."""
    report = f"""# Cultural Alignment System - Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report compares three approaches across {len(results)} test cases:
1. **Cultural Alignment System**: Uses 20-culture pool with smart expert selection
2. **Direct LLM**: Standard LLM response without cultural workflow
3. **Baseline Essay**: Generic essay-style response

## Aggregate Metrics

"""
    
    # Calculate aggregate metrics
    cultural_times = [r['cultural_alignment']['latency'] for r in results if 'latency' in r.get('cultural_alignment', {})]
    direct_times = [r['direct_llm']['latency'] for r in results if 'latency' in r.get('direct_llm', {})]
    baseline_times = [r['baseline']['latency'] for r in results if 'latency' in r.get('baseline', {})]
    
    cultural_lengths = [r['cultural_alignment']['response_length'] for r in results if 'response_length' in r.get('cultural_alignment', {})]
    direct_lengths = [r['direct_llm']['response_length'] for r in results if 'response_length' in r.get('direct_llm', {})]
    baseline_lengths = [r['baseline']['response_length'] for r in results if 'response_length' in r.get('baseline', {})]
    
    sensitive_count = sum(1 for r in results if r.get('cultural_alignment', {}).get('is_sensitive', False))
    
    report += f"""| Metric | Cultural Alignment | Direct LLM | Baseline |
|--------|-------------------|------------|----------|
| Avg Latency | {sum(cultural_times)/len(cultural_times):.1f}s | {sum(direct_times)/len(direct_times):.1f}s | {sum(baseline_times)/len(baseline_times):.1f}s |
| Avg Response Length | {sum(cultural_lengths)/len(cultural_lengths):.0f} chars | {sum(direct_lengths)/len(direct_lengths):.0f} chars | {sum(baseline_lengths)/len(baseline_lengths):.0f} chars |
| Culturally Sensitive | {sensitive_count}/{len(results)} | N/A | N/A |

## Detailed Test Results

"""
    
    # Add detailed results for each test
    for result in results:
        report += f"""### Test {result['test_id']}

**Question**: {result['question']}

**User**: {result['user_profile']['location']} (Age: {result['user_profile']['age']})

#### Cultural Alignment Response
"""
        
        if 'error' not in result.get('cultural_alignment', {}):
            ca = result['cultural_alignment']
            report += f"""- **Culturally Sensitive**: {'Yes' if ca['is_sensitive'] else 'No'} (Score: {ca['sensitivity_score']}/10)
- **Experts Consulted**: {ca['num_experts']}
- **Full Responses**: {', '.join(ca['full_responses']) if ca['full_responses'] else 'None'}
- **Brief Inputs**: {', '.join(ca['brief_responses']) if ca['brief_responses'] else 'None'}
- **Response Time**: {ca['latency']:.1f}s
- **Response Length**: {ca['response_length']} characters

**Response**:
> {ca['response'][:500]}...

**Cultural Insights**:
"""
            for insight in ca.get('cultural_insights', []):
                report += f"- {insight}\n"
        else:
            report += f"Error: {result['cultural_alignment']['error']}\n"
        
        report += f"""
#### Direct LLM Response
- **Response Time**: {result['direct_llm']['latency']:.1f}s
- **Response Length**: {result['direct_llm']['response_length']} characters

**Response**:
> {result['direct_llm']['response'][:500]}...

#### Baseline Essay Response
- **Response Time**: {result['baseline']['latency']:.1f}s
- **Response Length**: {result['baseline']['response_length']} characters

**Response**:
> {result['baseline']['response'][:500]}...

---

"""
    
    return report

def main():
    """Run the comparison benchmark."""
    print("\nStarting comprehensive comparison benchmark...")
    
    # Run tests
    results = run_comparison_benchmark(n_tests=3)  # Run 3 tests for demonstration
    
    # Generate report
    report = generate_comparison_report(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw JSON data
    json_path = f'/app/comparison_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'culture_pool_size': 20,
                'max_experts': 5,
                'approaches': ['cultural_alignment', 'direct_llm', 'baseline']
            },
            'results': results
        }, f, indent=2)
    
    # Save markdown report
    report_path = f'/app/comparison_report_{timestamp}.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  Report: {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    
    # Calculate summary statistics
    cultural_avg = sum(r['cultural_alignment']['latency'] for r in results if 'latency' in r.get('cultural_alignment', {})) / len(results)
    direct_avg = sum(r['direct_llm']['latency'] for r in results) / len(results)
    baseline_avg = sum(r['baseline']['latency'] for r in results) / len(results)
    
    print(f"Average Response Times:")
    print(f"  Cultural Alignment: {cultural_avg:.1f}s")
    print(f"  Direct LLM: {direct_avg:.1f}s")
    print(f"  Baseline: {baseline_avg:.1f}s")
    
    print(f"\nSpeedup of Direct LLM vs Cultural: {cultural_avg/direct_avg:.1f}x")
    print(f"Cultural system provides {sum(1 for r in results if r.get('cultural_alignment', {}).get('is_sensitive', False))}/{len(results)} culturally-aware responses")

if __name__ == "__main__":
    main()