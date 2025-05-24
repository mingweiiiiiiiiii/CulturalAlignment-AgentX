from typing import Dict, List, Tuple, Any
import concurrent.futures
import time
from llmagentsetting import llm_clients
from node.cultural_expert_node import CulturalExpertManager
from utility.measure_time import measure_time

@measure_time
def compose_final_response_parallel(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimized version that queries experts in parallel and uses caching.
    """
    # Get activate_set from router node
    activate_set: List[Tuple[str, float, str]] = state.get("activate_set", [])
    top_n = 3
    
    user_profile = state.get("user_profile", {})
    question_meta = state.get("question_meta", {})
    question = question_meta.get("original", "")
    sensitive_topics = question_meta.get("sensitive_topics", [])
    relevant_cultures = question_meta.get("relevant_cultures", [])
    
    # Select top-N experts by weight
    top_experts = sorted(activate_set, key=lambda x: x["weight"], reverse=True)[:top_n]
    
    # If no experts selected, provide a default response
    if not top_experts:
        final_response = "I'll provide a balanced perspective on your question, considering various cultural viewpoints."
        state["response_state"] = {
            "expert_responses": [],
            "final": final_response
        }
        return {"response_state": state["response_state"]}
    
    manager = CulturalExpertManager(state=state)
    manager.generate_expert_instances()
    
    # Parallel expert response generation
    def get_expert_response(entry):
        """Helper function to get response from one expert"""
        culture = entry["culture"]
        weight = entry["weight"]
        expert = manager.get_expert(culture)
        
        if not expert:
            return {
                "culture": culture,
                "weight": weight,
                "response": f"[Expert for {culture} not available]"
            }
        
        start = time.time()
        try:
            response_text = expert.generate_response(question)
            elapsed = time.time() - start
            print(f"Expert {culture} responded in {elapsed:.2f}s")
        except Exception as e:
            print(f"Error getting response from {culture} expert: {e}")
            response_text = f"[Error getting {culture} perspective]"
        
        return {
            "culture": culture,
            "weight": weight,
            "response": response_text
        }
    
    # Use ThreadPoolExecutor for parallel execution
    print(f"Querying {len(top_experts)} experts in parallel...")
    start_parallel = time.time()
    
    expert_responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(top_experts), 6)) as executor:
        future_to_expert = {
            executor.submit(get_expert_response, entry): entry 
            for entry in top_experts
        }
        
        for future in concurrent.futures.as_completed(future_to_expert):
            try:
                result = future.result()
                expert_responses.append(result)
            except Exception as e:
                entry = future_to_expert[future]
                print(f"Error with expert {entry['culture']}: {e}")
                expert_responses.append({
                    "culture": entry["culture"],
                    "weight": entry["weight"],
                    "response": f"[Error: {str(e)}]"
                })
    
    elapsed_parallel = time.time() - start_parallel
    print(f"All experts responded in {elapsed_parallel:.2f}s (parallel)")
    
    # Sort responses by weight to maintain consistency
    expert_responses = sorted(expert_responses, key=lambda x: x["weight"], reverse=True)
    
    # Create LLM prompt for final composition
    prompt_parts = [
        "You are a culturally-aware assistant tasked with composing a final response that is sensitive to the user's background and preferences.",
        "Please synthesize the expert responses and produce a final answer that is coherent, respectful, and **under 200 words**.",
        f"Question: {question}",
        "\nUser Profile:",
        f"- Demographics and preferences: {user_profile}",
        "\nCulturally Diverse Expert Responses:",
    ]
    
    for i, resp in enumerate(expert_responses, 1):
        prompt_parts.append(f"{i}. ({resp['culture']}, weight: {resp['weight']:.2f}) {resp['response']}")
    
    if sensitive_topics:
        prompt_parts.append(f"\nSensitive Topics: {', '.join(sensitive_topics)}")
    if relevant_cultures:
        prompt_parts.append(f"Relevant Cultures: {', '.join(relevant_cultures)}")
    
    prompt_parts.append(
        "\nReminder: Keep your final response within 200 words, blending insights thoughtfully but briefly."
    )
    
    llm_prompt = "\n".join(prompt_parts)
    
    # Generate final composed response
    client = llm_clients.LambdaAPIClient(state=state)
    try:
        final_response = client.get_completion(llm_prompt, temperature=0.7, max_tokens=300)
    except Exception as e:
        final_response = f"[LLM Error: {str(e)}]"
    
    # Optional soft enforcement (post-process if needed)
    words = final_response.split()
    if len(words) > 200:
        final_response = " ".join(words[:200]) + "..."
    
    # Update response state
    response_state = state.get("response_state", {})
    response_state.update({
        "expert_responses": expert_responses,
        "final": final_response,
        "parallel_time": elapsed_parallel
    })
    
    state["activate_compose"] = False
    
    return {"response_state": response_state}