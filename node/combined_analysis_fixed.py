from typing import Dict, List, Tuple
from llmagentsetting.ollama_client import OllamaClient
from utility.measure_time import measure_time
import re
import json

@measure_time
def analyze_question_sensitivity(state: Dict) -> Dict:
    """
    Combined node that performs both sensitivity check and topic extraction in a single LLM call.
    This reduces the number of sequential LLM calls in the workflow.
    """
    question_meta = state.get("question_meta", {})
    question = question_meta.get("original", "")
    
    # Single prompt that asks for both sensitivity analysis and topic extraction
    combined_prompt = f"""Analyze the following question for cultural sensitivity and extract relevant topics.

Question: {question}

You must provide your analysis in valid JSON format. Here is the required structure:
{{
    "sensitivity_score": <number from 0-10>,
    "is_sensitive": <true or false>,
    "sensitive_topics": ["topic1", "topic2"],
    "relevant_cultures": ["culture1", "culture2"],
    "reasoning": "brief explanation"
}}

Guidelines:
- sensitivity_score: 0-10 where 0 is not sensitive and 10 is extremely sensitive
- is_sensitive: true if score >= 5, false otherwise
- sensitive_topics: list of culturally sensitive topics mentioned
- relevant_cultures: list of cultures that might have different perspectives
- reasoning: brief explanation of your sensitivity assessment

Consider factors like:
- Political systems and governance
- Religious or spiritual beliefs
- Cultural traditions and practices
- Economic systems and values
- Social norms and customs
- Historical conflicts or tensions

IMPORTANT: Respond ONLY with valid JSON, no additional text."""

    # Get LLM response using local ollama
    client = OllamaClient()
    try:
        response = client.generate(combined_prompt)
        
        # Try to parse JSON response
        # First, try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            # Fallback parsing if JSON extraction fails
            analysis = {
                "sensitivity_score": 5,
                "is_sensitive": True,
                "sensitive_topics": [],
                "relevant_cultures": [],
                "reasoning": response[:100] if response else "Unable to analyze"
            }
            
            # Try to extract score from text
            score_match = re.search(r'score[:\s]+(\d+)', response, re.IGNORECASE)
            if score_match:
                analysis["sensitivity_score"] = int(score_match.group(1))
                analysis["is_sensitive"] = analysis["sensitivity_score"] >= 5
        
        # Update state with combined results
        question_meta.update({
            "sensitivity_score": analysis.get("sensitivity_score", 5),
            "is_sensitive": analysis.get("is_sensitive", True),
            "sensitive_topics": analysis.get("sensitive_topics", []),
            "relevant_cultures": analysis.get("relevant_cultures", []),
            "sensitivity_reasoning": analysis.get("reasoning", "")
        })
        
        # Set workflow flags based on sensitivity
        state["is_sensitive"] = analysis.get("is_sensitive", True)
        state["topics_extracted"] = True  # Mark as done since we extracted topics too
        
        # Determine next step
        if state["is_sensitive"]:
            state["activate_router"] = True
        else:
            state["activate_compose"] = True
            
    except Exception as e:
        print(f"Error in combined analysis: {e}")
        # Fallback to conservative defaults
        question_meta.update({
            "sensitivity_score": 5,
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