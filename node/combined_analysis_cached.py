from typing import Dict
from node.combined_analysis_fixed import analyze_question_sensitivity as original_analyze
from utility.cache_manager import get_cache_manager
from utility.measure_time import measure_time

@measure_time
def analyze_question_sensitivity_cached(state: Dict) -> Dict:
    """
    Cached version of combined sensitivity + topic extraction.
    Checks cache first, falls back to original if not found.
    """
    question_meta = state.get("question_meta", {})
    question = question_meta.get("original", "")
    
    # Check cache first
    cache = get_cache_manager()
    cached_result = cache.get_sensitivity_analysis(question)
    
    if cached_result:
        print(f"✅ Cache hit for sensitivity analysis")
        # Update state with cached results
        question_meta.update(cached_result)
        state["is_sensitive"] = cached_result.get("is_sensitive", True)
        state["topics_extracted"] = True
        
        # Determine next step
        if state["is_sensitive"]:
            state["activate_router"] = True
        else:
            state["activate_compose"] = True
            
        return {
            "question_meta": question_meta,
            "is_sensitive": state["is_sensitive"],
            "topics_extracted": True
        }
    
    print(f"❌ Cache miss for sensitivity analysis")
    # Call original function
    result = original_analyze(state)
    
    # Cache the results
    analysis_data = {
        "sensitivity_score": state["question_meta"].get("sensitivity_score", 5),
        "is_sensitive": state["question_meta"].get("is_sensitive", True),
        "sensitive_topics": state["question_meta"].get("sensitive_topics", []),
        "relevant_cultures": state["question_meta"].get("relevant_cultures", []),
        "sensitivity_reasoning": state["question_meta"].get("sensitivity_reasoning", "")
    }
    cache.put_sensitivity_analysis(question, analysis_data)
    
    return result