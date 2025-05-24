from typing import Dict, List, Tuple
from llmagentsetting import llm_clients
from utility.measure_time import measure_time
import re
import json

@measure_time
def analyze_question_sensitivity_enhanced(state: Dict) -> Dict:
    """
    Enhanced sensitivity analysis with topic-specific boosting and cultural context awareness.
    Addresses the threshold issues identified in testing.
    """
    question_meta = state.get("question_meta", {})
    user_profile = state.get("user_profile", {})
    question = question_meta.get("original", "")
    
    # Extract cultural context from persona
    ancestry = user_profile.get("ancestry", "")
    country = user_profile.get("country", "")
    age = user_profile.get("age", "")
    
    # Enhanced prompt with explicit cultural context
    enhanced_prompt = f"""Analyze the following question for cultural sensitivity, considering the respondent's cultural background.

Respondent Profile: {ancestry} from {country}, age {age}

Question: {question}

Cultural Context Analysis:
Consider how different cultures might approach this topic differently. Pay special attention to:
- Educational values and parental roles (varies significantly across cultures)
- Work ethics and dedication expectations (East Asian vs Western differences)
- Family obligations and social hierarchies (collectivist vs individualist cultures)
- Religious and traditional practices
- Authority, respect, and generational relationships

Scoring Guidelines:
- 0-2: Universal agreement across cultures
- 3-4: Some cultural variation exists
- 5-6: Moderate cultural differences likely
- 7-8: Significant cultural divisions expected
- 9-10: Major cultural conflicts or sensitivities

Provide your analysis in JSON format:
{{
    "sensitivity_score": <number from 0-10>,
    "is_sensitive": <true if score >= 4, false otherwise>,
    "sensitive_topics": [<list of culturally sensitive topics mentioned>],
    "relevant_cultures": [<list of cultures that might have different perspectives>],
    "reasoning": "<brief explanation of cultural sensitivity factors>",
    "cultural_factors": [<specific cultural dimensions that create variation>]
}}

Be more sensitive to detecting cultural variation, especially for topics involving:
- Education and parental authority
- Work culture and dedication
- Family obligations and living arrangements
- Social hierarchy and respect for elders
- Religious practices and traditions"""

    # Get LLM response
    client = llm_clients.LambdaAPIClient(state=state)
    try:
        response = client.get_completion(enhanced_prompt, temperature=0.3, max_tokens=400)
        
        # Try to parse JSON response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            # Fallback parsing
            analysis = {
                "sensitivity_score": 5,
                "is_sensitive": True,
                "sensitive_topics": [],
                "relevant_cultures": [],
                "reasoning": response,
                "cultural_factors": []
            }
            
            # Try to extract score from text
            score_match = re.search(r'score[:\s]+(\d+)', response, re.IGNORECASE)
            if score_match:
                analysis["sensitivity_score"] = int(score_match.group(1))
        
        # Apply topic-specific and cultural context boosts
        base_score = analysis.get("sensitivity_score", 5)
        boosted_score = apply_sensitivity_boosts(base_score, question, user_profile)
        
        # Update analysis with boosted score
        analysis["sensitivity_score"] = boosted_score
        analysis["is_sensitive"] = boosted_score >= 4  # Lower threshold
        
        # Update state with results
        question_meta.update({
            "sensitivity_score": boosted_score,
            "is_sensitive": analysis.get("is_sensitive", True),
            "sensitive_topics": analysis.get("sensitive_topics", []),
            "relevant_cultures": analysis.get("relevant_cultures", []),
            "sensitivity_reasoning": analysis.get("reasoning", ""),
            "cultural_factors": analysis.get("cultural_factors", [])
        })
        
        # Set workflow flags
        state["is_sensitive"] = analysis.get("is_sensitive", True)
        state["topics_extracted"] = True
        
        # Determine next step
        if state["is_sensitive"]:
            state["activate_router"] = True
        else:
            state["activate_compose"] = True
            
    except Exception as e:
        print(f"Error in enhanced sensitivity analysis: {e}")
        # Conservative fallback
        question_meta.update({
            "sensitivity_score": 6,  # Conservative default
            "is_sensitive": True,
            "sensitive_topics": [],
            "relevant_cultures": [],
            "sensitivity_reasoning": f"Analysis error: {str(e)}"
        })
        state["is_sensitive"] = True
        state["activate_router"] = True
    
    return {
        "question_meta": question_meta,
        "is_sensitive": state["is_sensitive"],
        "topics_extracted": state.get("topics_extracted", True)
    }

def apply_sensitivity_boosts(base_score: int, question: str, user_profile: Dict) -> int:
    """
    Apply topic-specific and cultural context boosts to improve sensitivity detection.
    """
    boosted_score = base_score
    question_lower = question.lower()
    ancestry = user_profile.get("ancestry", "").lower()
    
    # Education and family topic boost (+2)
    education_keywords = ["education", "school", "parent", "children", "family role", "teaching", "learning"]
    if any(keyword in question_lower for keyword in education_keywords):
        boosted_score += 2
        print(f"Applied education topic boost: +2 (base: {base_score} → {boosted_score})")
    
    # Social hierarchy and respect boost (+2)
    hierarchy_keywords = ["elder", "respect", "authority", "hierarchy", "younger", "older", "traditional", "honor"]
    if any(keyword in question_lower for keyword in hierarchy_keywords):
        boosted_score += 2
        print(f"Applied hierarchy topic boost: +2 (base: {base_score} → {boosted_score})")
    
    # Work culture boost (+1)
    work_keywords = ["work", "dedication", "hours", "career", "job", "professional", "workplace"]
    if any(keyword in question_lower for keyword in work_keywords):
        boosted_score += 1
        print(f"Applied work culture boost: +1 (base: {base_score} → {boosted_score})")
    
    # Strong cultural context boost (+1)
    strong_cultural_contexts = ["japanese", "korean", "chinese", "indian", "middle eastern", "hispanic", "asian"]
    if any(culture in ancestry for culture in strong_cultural_contexts):
        boosted_score += 1
        print(f"Applied cultural context boost for {ancestry}: +1")
    
    # Religious and traditional practices boost (+2)
    religious_keywords = ["religious", "tradition", "practice", "spiritual", "faith", "belief", "ritual"]
    if any(keyword in question_lower for keyword in religious_keywords):
        boosted_score += 2
        print(f"Applied religious/traditional boost: +2")
    
    # Cap at maximum score
    final_score = min(10, boosted_score)
    
    if final_score != base_score:
        print(f"Total sensitivity boost applied: {base_score} → {final_score}")
    
    return final_score