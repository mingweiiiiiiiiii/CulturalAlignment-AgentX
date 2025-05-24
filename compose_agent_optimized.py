from typing import Dict, List, Any
import concurrent.futures
import time
from llmagentsetting import llm_clients
from node.cultural_expert_node import CulturalExpertManager
from utility.measure_time import measure_time

@measure_time
def compose_final_response_optimized(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimized version that queries experts in parallel
    """
    # Get activate_set from router node
    activate_set = state.get("activate_set", [])
    top_n = 3
    
    question_meta = state.get("question_meta", {})
    question = question_meta.get("original", "")
    
    # Select top-N experts by weight
    top_experts = sorted(activate_set, key=lambda x: x["weight"], reverse=True)[:top_n]
    
    manager = CulturalExpertManager(state=state)
    manager.generate_expert_instances()
    
    # Parallel expert response generation
    expert_responses = []
    
    def get_expert_response(entry):
        """Helper function to get response from one expert"""
        culture = entry["culture"]
        weight = entry["weight"]
        expert = manager.get_expert(culture)
        
        start = time.time()
        response_text = expert.generate_response(question)
        elapsed = time.time() - start
        
        print(f"Expert {culture} responded in {elapsed:.2f}s")
        
        return {
            "culture": culture,
            "weight": weight,
            "response": response_text
        }
    
    # Use ThreadPoolExecutor for parallel execution
    print(f"Querying {len(top_experts)} experts in parallel...")
    start_parallel = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_expert = {
            executor.submit(get_expert_response, entry): entry 
            for entry in top_experts
        }
        
        for future in concurrent.futures.as_completed(future_to_expert):
            try:
                result = future.result()
                expert_responses.append(result)
            except Exception as exc:
                entry = future_to_expert[future]
                print(f"Expert {entry['culture']} generated an exception: {exc}")
                expert_responses.append({
                    "culture": entry["culture"],
                    "weight": entry["weight"],
                    "response": f"Error: {str(exc)}"
                })
    
    parallel_time = time.time() - start_parallel
    print(f"All experts responded in {parallel_time:.2f}s (parallel)")
    
    # Update state with expert responses
    state["response_state"]["expert_responses"] = expert_responses
    
    # Rest of the composition logic would go here...
    # For now, just concatenate responses
    final_response = "\n\n".join([
        f"[{r['culture']} perspective]:\n{r['response']}" 
        for r in expert_responses
    ])
    
    state["response_state"]["final"] = final_response
    state["activate_compose"] = False
    
    return state