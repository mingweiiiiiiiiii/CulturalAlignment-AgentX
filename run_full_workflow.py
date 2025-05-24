"""
Run a full cultural alignment workflow demonstration.
"""
import sys
sys.path.append('/app')

from mylanggraph.graph_optimized import create_optimized_cultural_graph
from datetime import datetime
import json

def run_cultural_analysis():
    """Run a complete cultural analysis workflow."""
    
    # Test question and user profile
    question = "Should women prioritize career advancement or family responsibilities?"
    user_profile = {
        'age': 35,
        'gender': 'female',
        'location': 'Japan',
        'cultural_background': 'East Asian',
        'values': ['family harmony', 'professional growth', 'respect for tradition']
    }
    
    # Initialize state
    state = {
        'question_meta': {
            'original': question,
            'rephrased': '',
            'language': 'en',
            'timestamp': datetime.now().isoformat()
        },
        'user_profile': user_profile,
        'is_sensitive': False,
        'activate_sensitivity_check': True,
        'activate_extract_topics': False,
        'activate_router': False,
        'activate_compose': False,
        'current_state': 'sensitivity_check',
        '__start__': 'sensitivity_check',
        'steps': [],
        'expert_consultations': {},
        'final_response': {}
    }
    
    print("=" * 80)
    print("CULTURAL ALIGNMENT AGENT - FULL WORKFLOW DEMONSTRATION")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"\nUser Profile:")
    print(f"  - Age: {user_profile['age']}")
    print(f"  - Gender: {user_profile['gender']}")
    print(f"  - Location: {user_profile['location']}")
    print(f"  - Cultural Background: {user_profile['cultural_background']}")
    print(f"  - Values: {', '.join(user_profile['values'])}")
    print("-" * 80)
    
    # Create the optimized graph
    print("\nInitializing cultural alignment graph...")
    graph = create_optimized_cultural_graph()
    
    # Run the workflow
    print("Running analysis workflow...\n")
    try:
        # Configure with thread_id for checkpointing
        config = {"configurable": {"thread_id": "demo_run_001"}}
        result = graph.invoke(state, config=config)
        
        # Display results
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        
        # Extract metadata
        meta = result.get('question_meta', {})
        
        # 1. Sensitivity Analysis
        print("\n1. SENSITIVITY ANALYSIS")
        print("-" * 40)
        print(f"Is Culturally Sensitive: {meta.get('is_sensitive', 'Unknown')}")
        print(f"Sensitivity Score: {meta.get('sensitivity_score', 'N/A')}/10")
        print(f"Reasoning: {meta.get('sensitivity_reasoning', 'Not available')}")
        
        # 2. Cultural Topics
        if meta.get('sensitive_topics'):
            print(f"\nSensitive Topics Identified:")
            for topic in meta.get('sensitive_topics', []):
                print(f"  • {topic}")
                
        if meta.get('relevant_cultures'):
            print(f"\nRelevant Cultural Perspectives:")
            for culture in meta.get('relevant_cultures', []):
                print(f"  • {culture}")
        
        # 3. Expert Consultations
        experts = result.get('expert_consultations', {})
        if experts:
            print(f"\n2. EXPERT CONSULTATIONS ({len(experts)} experts)")
            print("-" * 40)
            for expert_name, consultation in experts.items():
                print(f"\n{expert_name.upper()} Expert:")
                # Show first 300 chars of each expert response
                response_preview = consultation[:300] + "..." if len(consultation) > 300 else consultation
                print(f"{response_preview}")
        
        # 4. Final Response
        final = result.get('final_response', {})
        if final:
            print("\n3. FINAL CULTURALLY-ALIGNED RESPONSE")
            print("-" * 40)
            
            main_response = final.get('main_response', 'No response generated')
            print(f"\nMain Response:")
            print(f"{main_response}")
            
            if final.get('cultural_considerations'):
                print(f"\nCultural Considerations:")
                for i, consideration in enumerate(final.get('cultural_considerations', []), 1):
                    print(f"  {i}. {consideration}")
            
            if final.get('disclaimer'):
                print(f"\nDisclaimer: {final.get('disclaimer')}")
        
        # 5. Workflow Steps
        steps = result.get('steps', [])
        if steps:
            print("\n4. WORKFLOW EXECUTION STEPS")
            print("-" * 40)
            for i, step in enumerate(steps, 1):
                print(f"  Step {i}: {step}")
        
        # Save detailed results
        with open('/app/workflow_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'user_profile': user_profile,
                'results': {
                    'sensitivity_analysis': {
                        'is_sensitive': meta.get('is_sensitive'),
                        'score': meta.get('sensitivity_score'),
                        'reasoning': meta.get('sensitivity_reasoning'),
                        'topics': meta.get('sensitive_topics', []),
                        'cultures': meta.get('relevant_cultures', [])
                    },
                    'expert_consultations': experts,
                    'final_response': final,
                    'workflow_steps': steps
                }
            }, f, indent=2)
        
        print(f"\n\nDetailed results saved to: /app/workflow_results.json")
        
    except Exception as e:
        print(f"\nError during workflow execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_cultural_analysis()