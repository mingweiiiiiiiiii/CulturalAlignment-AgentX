"""
Simplified workflow runner using local ollama client throughout.
"""
import sys
sys.path.append('/app')

from typing import Dict, List
from datetime import datetime
from llmagentsetting.ollama_client import OllamaClient
from node.sensitivity_optimized import analyze_question_sensitivity
# from node.embed_utils import embed_text  # Not needed for simplified version
import json
import time

def get_cultural_expert_response(culture: str, question: str, client: OllamaClient) -> str:
    """Get response from a cultural expert."""
    prompt = f"""You are a cultural expert from {culture}, deeply familiar with its historical, social, moral, and traditional nuances.
Frame your answer considering the values, etiquette, common beliefs, communication styles, and societal norms typical of {culture}.
Include aspects like community vs individualism, indirect vs direct communication, formality levels, views on authority, spirituality, family roles, and social relationships.
Be thoughtful, factual, respectful of diversity, and avoid generalizations or stereotypes.
Keep the response under 150 words.

Question: '{question}'"""
    
    return client.generate(prompt)

def compose_final_response(question: str, expert_responses: Dict[str, str], user_profile: Dict, client: OllamaClient) -> Dict:
    """Compose final culturally-aligned response."""
    
    # Build synthesis prompt
    expert_summaries = "\n\n".join([
        f"{culture} perspective:\n{response}"
        for culture, response in expert_responses.items()
    ])
    
    prompt = f"""You are tasked with providing a culturally sensitive and balanced response to a question, considering multiple cultural perspectives.

User Profile:
- Location: {user_profile.get('location', 'Unknown')}
- Cultural Background: {user_profile.get('cultural_background', 'Unknown')}
- Age: {user_profile.get('age', 'Unknown')}
- Gender: {user_profile.get('gender', 'Unknown')}

Question: {question}

Cultural Expert Perspectives:
{expert_summaries}

Please synthesize these perspectives into a balanced, respectful response that:
1. Acknowledges the cultural sensitivity of the topic
2. Presents multiple viewpoints fairly
3. Is particularly mindful of the user's cultural background
4. Provides practical guidance where appropriate
5. Remains under 200 words

Response:"""
    
    main_response = client.generate(prompt)
    
    # Extract key cultural considerations
    considerations_prompt = f"""Based on the following culturally sensitive response, list 3-5 key cultural considerations in brief bullet points:

Response: {main_response}

List only the cultural considerations, one per line:"""
    
    considerations_text = client.generate(considerations_prompt)
    considerations = [line.strip().lstrip('- •') for line in considerations_text.split('\n') if line.strip() and line.strip() not in ['Cultural Considerations:', 'Key Cultural Considerations:']][:5]
    
    return {
        "main_response": main_response,
        "cultural_considerations": considerations,
        "disclaimer": "This response considers multiple cultural perspectives. Individual views may vary within each culture."
    }

def run_simplified_workflow():
    """Run a simplified but complete cultural workflow."""
    
    # Test scenario
    question = "Should women prioritize career advancement or family responsibilities?"
    user_profile = {
        'age': 35,
        'gender': 'female',
        'location': 'Japan',
        'cultural_background': 'East Asian',
        'values': ['family harmony', 'professional growth', 'respect for tradition']
    }
    
    print("=" * 80)
    print("CULTURAL ALIGNMENT WORKFLOW - SIMPLIFIED DEMONSTRATION")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"\nUser Profile:")
    for key, value in user_profile.items():
        if isinstance(value, list):
            print(f"  - {key}: {', '.join(value)}")
        else:
            print(f"  - {key}: {value}")
    print("-" * 80)
    
    # Initialize
    client = OllamaClient()
    workflow_steps = []
    start_time = time.time()
    
    # Step 1: Sensitivity Analysis
    print("\n1. ANALYZING CULTURAL SENSITIVITY...")
    state = {
        "question_meta": {"original": question},
        "user_profile": user_profile
    }
    
    sensitivity_result = analyze_question_sensitivity(state)
    meta = sensitivity_result["question_meta"]
    workflow_steps.append(f"Sensitivity analysis completed (score: {meta['sensitivity_score']}/10)")
    
    print(f"   ✓ Sensitivity Score: {meta['sensitivity_score']}/10")
    print(f"   ✓ Is Sensitive: {meta['is_sensitive']}")
    print(f"   ✓ Reasoning: {meta['sensitivity_reasoning'][:100]}...")
    
    # Step 2: Select Cultural Experts (if sensitive)
    expert_responses = {}
    if meta['is_sensitive']:
        print("\n2. CONSULTING CULTURAL EXPERTS...")
        
        # Select diverse cultures for consultation
        cultures_to_consult = ["United States", "Japan", "India"]
        
        # Add user's culture if not already included
        user_culture = user_profile.get('location', '')
        if user_culture and user_culture not in cultures_to_consult:
            cultures_to_consult[1] = user_culture  # Replace middle culture
        
        # Get expert responses
        for culture in cultures_to_consult:
            print(f"   Consulting {culture} expert...", end='', flush=True)
            try:
                response = get_cultural_expert_response(culture, question, client)
                expert_responses[culture] = response
                workflow_steps.append(f"Consulted {culture} cultural expert")
                print(" ✓")
            except Exception as e:
                print(f" ✗ Error: {e}")
                expert_responses[culture] = "Unable to generate response."
    
    # Step 3: Compose Final Response
    print("\n3. COMPOSING CULTURALLY-ALIGNED RESPONSE...")
    if expert_responses:
        final_response = compose_final_response(question, expert_responses, user_profile, client)
        workflow_steps.append("Composed final culturally-aligned response")
    else:
        # Non-sensitive question - direct response
        prompt = f"Please provide a helpful response to this question: {question}"
        main_response = client.generate(prompt)
        final_response = {
            "main_response": main_response,
            "cultural_considerations": [],
            "disclaimer": ""
        }
        workflow_steps.append("Generated direct response (non-sensitive question)")
    
    print("   ✓ Response composed")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Display Results
    print("\n" + "=" * 80)
    print("COMPLETE RESULTS")
    print("=" * 80)
    
    print("\n📊 SENSITIVITY ANALYSIS:")
    print(f"   Score: {meta['sensitivity_score']}/10")
    print(f"   Culturally Sensitive: {'Yes' if meta['is_sensitive'] else 'No'}")
    if meta.get('sensitive_topics'):
        print(f"   Topics: {', '.join(meta['sensitive_topics'])}")
    
    if expert_responses:
        print(f"\n👥 EXPERT CONSULTATIONS ({len(expert_responses)} experts):")
        for culture, response in expert_responses.items():
            print(f"\n   {culture.upper()}:")
            print(f"   {response[:150]}..." if len(response) > 150 else f"   {response}")
    
    print("\n💬 FINAL RESPONSE:")
    print("-" * 40)
    print(final_response['main_response'])
    
    if final_response['cultural_considerations']:
        print("\n🌍 CULTURAL CONSIDERATIONS:")
        for i, consideration in enumerate(final_response['cultural_considerations'], 1):
            print(f"   {i}. {consideration}")
    
    if final_response['disclaimer']:
        print(f"\n⚠️  {final_response['disclaimer']}")
    
    print(f"\n⏱️  Total processing time: {total_time:.1f} seconds")
    
    print("\n📋 WORKFLOW STEPS:")
    for i, step in enumerate(workflow_steps, 1):
        print(f"   {i}. {step}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "user_profile": user_profile,
        "sensitivity_analysis": {
            "score": meta['sensitivity_score'],
            "is_sensitive": meta['is_sensitive'],
            "reasoning": meta['sensitivity_reasoning'],
            "topics": meta.get('sensitive_topics', [])
        },
        "expert_consultations": expert_responses,
        "final_response": final_response,
        "workflow_steps": workflow_steps,
        "processing_time": total_time
    }
    
    with open('/app/simplified_workflow_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: /app/simplified_workflow_results.json")

if __name__ == "__main__":
    run_simplified_workflow()