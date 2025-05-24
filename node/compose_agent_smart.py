"""
Smart composition agent that handles both full and brief cultural responses.
"""
from typing import Dict, List
from llmagentsetting.ollama_client import OllamaClient
from utility.measure_time import measure_time

@measure_time
def compose_final_response_smart(state: Dict) -> Dict:
    """
    Compose final response from mix of full and brief cultural inputs.
    Prioritizes full responses but incorporates brief perspectives.
    """
    question = state["question_meta"]["original"]
    user_profile = state.get("user_profile", {})
    expert_responses = state.get("expert_responses", {})
    
    if not expert_responses:
        # No expert responses - provide direct answer
        return compose_direct_response(state)
    
    # Separate full and brief responses
    full_responses = {}
    brief_responses = {}
    
    for culture, info in expert_responses.items():
        if info['response_type'] == 'full':
            full_responses[culture] = info
        else:
            brief_responses[culture] = info
    
    print(f"Composing from {len(full_responses)} full and {len(brief_responses)} brief responses")
    
    # Build synthesis prompt
    client = OllamaClient()
    
    # Start with full responses
    full_section = ""
    if full_responses:
        full_section = "DETAILED CULTURAL PERSPECTIVES:\n\n"
        for culture, info in full_responses.items():
            full_section += f"{culture} (Relevance: {info['relevance_score']}/10):\n"
            full_section += f"{info['response']}\n\n"
    
    # Add brief responses if any
    brief_section = ""
    if brief_responses:
        brief_section = "\nADDITIONAL CULTURAL NOTES:\n"
        for culture, info in brief_responses.items():
            if info['response'] and len(info['response']) > 10:  # Skip empty responses
                brief_section += f"- {culture}: {info['response']}\n"
    
    # Synthesis prompt
    prompt = f"""You are synthesizing multiple cultural perspectives into a balanced response.

Question: {question}

User Profile:
- Location: {user_profile.get('location', 'Unknown')}
- Cultural Background: {user_profile.get('cultural_background', 'Unknown')}

{full_section}
{brief_section}

Please provide a culturally-aware response that:
1. Prioritizes the detailed perspectives from cultures where this topic is most relevant
2. Acknowledges additional viewpoints from the brief notes
3. Is sensitive to the user's cultural background ({user_profile.get('location', 'Unknown')})
4. Remains balanced and respectful of all perspectives
5. Is concise (under 200 words)

Response:"""

    try:
        final_response = client.generate(prompt)
        
        # Generate cultural insights
        insights = generate_cultural_insights(full_responses, brief_responses, user_profile)
        
        state["final_response"] = {
            "main_response": final_response,
            "cultural_insights": insights,
            "num_full_perspectives": len(full_responses),
            "num_brief_perspectives": len(brief_responses),
            "primary_cultures": list(full_responses.keys()),
            "additional_cultures": list(brief_responses.keys())
        }
        
    except Exception as e:
        print(f"Error composing response: {e}")
        state["final_response"] = {
            "main_response": "Unable to generate culturally-aligned response.",
            "error": str(e)
        }
    
    return state

def compose_direct_response(state: Dict) -> Dict:
    """Fallback for non-sensitive questions."""
    question = state["question_meta"]["original"]
    client = OllamaClient()
    
    prompt = f"""Please provide a helpful response to this question:

{question}

Give a balanced, informative answer:"""
    
    try:
        response = client.generate(prompt)
        state["final_response"] = {
            "main_response": response,
            "cultural_insights": ["This topic has universal relevance across cultures."],
            "num_full_perspectives": 0,
            "num_brief_perspectives": 0
        }
    except Exception as e:
        state["final_response"] = {
            "main_response": "Unable to generate response.",
            "error": str(e)
        }
    
    return state

def generate_cultural_insights(full_responses: Dict, brief_responses: Dict, 
                              user_profile: Dict) -> List[str]:
    """Generate key cultural insights from the responses."""
    insights = []
    
    # Insight about primary cultural perspectives
    if full_responses:
        primary_cultures = list(full_responses.keys())
        insights.append(f"Primary cultural perspectives from: {', '.join(primary_cultures)}")
    
    # Insight about consensus or divergence
    if len(full_responses) >= 2:
        # Check for consensus themes (simplified - could use NLP)
        insights.append("Multiple cultures show varying approaches to this topic")
    
    # Insight about user's culture
    user_location = user_profile.get('location')
    if user_location and user_location in full_responses:
        insights.append(f"Your culture ({user_location}) has specific perspectives on this topic")
    elif user_location and user_location in brief_responses:
        insights.append(f"This topic has limited specific relevance to {user_location} culture")
    
    # Insight about cultural sensitivity
    avg_relevance = sum(r['relevance_score'] for r in full_responses.values()) / len(full_responses) if full_responses else 0
    if avg_relevance >= 7:
        insights.append("This is a highly culturally sensitive topic with significant variations")
    elif avg_relevance >= 5:
        insights.append("This topic shows moderate cultural variation")
    
    return insights

if __name__ == "__main__":
    # Test the smart composer
    test_state = {
        "question_meta": {
            "original": "What are your views on gender roles in the family?"
        },
        "user_profile": {
            "location": "Japan",
            "cultural_background": "East Asian"
        },
        "expert_responses": {
            "Japan": {
                "response": "In Japan, traditional gender roles have been evolving...",
                "relevance_score": 8,
                "response_type": "full"
            },
            "India": {
                "response": "Indian perspectives on family gender roles vary greatly...",
                "relevance_score": 7,
                "response_type": "full"
            },
            "Germany": {
                "response": "Generally egalitarian approach",
                "relevance_score": 4,
                "response_type": "brief"
            },
            "Brazil": {
                "response": "Mix of traditional and modern views",
                "relevance_score": 3,
                "response_type": "brief"
            }
        }
    }
    
    print("Testing smart composer...")
    result = compose_final_response_smart(test_state)
    
    print("\nFinal response generated:")
    print(f"Full perspectives used: {result['final_response']['num_full_perspectives']}")
    print(f"Brief perspectives used: {result['final_response']['num_brief_perspectives']}")
    print(f"Cultural insights: {result['final_response']['cultural_insights']}")