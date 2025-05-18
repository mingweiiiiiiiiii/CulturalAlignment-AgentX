from typing import Dict, List, Tuple, Any
import google.generativeai as genai
from llmagentsetting import llm_clients
from node.cultural_expert_node import CulturalExpertManager
from utility.measure_time import measure_time

# Define ExpertResponse type
ExpertResponse = Dict[str, str]
#  Compose Final Response
@measure_time
def compose_final_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compose a final response based on multiple cultural expert inputs,
    aiming for a concise culturally-informed answer within 200 words.
    """

    # Step 0: Get activate_set from router node
    activate_set: List[Tuple[str, float, str]] = state.get("activate_set", [])
    top_n = 3

    user_profile = state.get("user_profile", {})
    question_meta = state.get("question_meta", {})
    preferences = user_profile.get("preferences", {})
    demographics = user_profile  # since separate demographic and preferences info is not provided

    question = question_meta.get("original", "")
    sensitive_topics = question_meta.get("sensitive_topics", [])
    relevant_cultures = question_meta.get("relevant_cultures", [])
    # Step 1: Select top-N experts by weight
    top_experts = sorted(activate_set, key=lambda x: x["weight"], reverse=True)[:top_n]

    manager = CulturalExpertManager(state=state)
    manager.generate_expert_instances() 
    expert_responses = []
    for entry in top_experts:
        culture = entry["culture"]
        weight  = entry["weight"]
        expert  = manager.get_expert(culture)              # retrieve the expert instance
        response_text = expert.generate_response(question)  # only K LLM calls now
        expert_responses.append({
            "culture": culture,
            "weight": weight,
            "response": response_text
        })

    # Step 2: Create LLM prompt

    prompt_parts = [
        "You are a culturally-aware assistant tasked with composing a final response that is sensitive to the user's background and preferences.",
        "Please synthesize the expert responses and produce a final answer that is coherent, respectful, and **under 200 words**.",
        f"Question: {question}",
        "\nUser Profile:",
        f"- Demographics and preferences: {demographics}",
       # f"- Preferences: {preferences}",
        "\nCulturally Diverse Expert Responses:",
    ]

    for i, resp in enumerate(expert_responses, 1):
        prompt_parts.append(f"{i}. ({resp['culture']}) {resp['response']}")

    if sensitive_topics:
        prompt_parts.append(f"\nSensitive Topics: {', '.join(sensitive_topics)}")
    if relevant_cultures:
        prompt_parts.append(f"Relevant Cultures: {', '.join(relevant_cultures)}")

    prompt_parts.append(
        "\nReminder: Keep your final response within 200 words, blending insights thoughtfully but briefly."
    )

    llm_prompt = "\n".join(prompt_parts)

    # Step 3: Generate final composed response
    # GoogleStudio_API_KEY = "AIzaSyAlMLq2h1YHKJgOm6hds2aHz_iWrByXacM"
    # genai.configure(api_key=GoogleStudio_API_KEY)
    client = llm_clients.LambdaAPIClient(state=state)
    try:
        response = client.get_completion(llm_prompt)
        final_response = response
    except Exception as e:
        final_response = f"[LLM Error: {str(e)}]"
    
    # Step 4: Optional soft enforcement (post-process if needed)
    words = final_response.split()
    if len(words) > 200:
        final_response = " ".join(words[:200]) + "..."

    # Step 5: Update response state
    response_state = state.get("response_state", {})
    response_state.update(
        {
            "expert_responses": expert_responses,
            "final": final_response
        }
    )
    state["activate_compose"] = False
    return {
        "response_state": response_state,
    }

# ===============================
# 🧪 Test Case
# ===============================
if __name__ == "__main__":
    pass

    # Dummy GraphState simulation
    state = GraphState(
        {
            "user_profile": {
                "preferences": {"language": "English"},
                "demographics": {"age": 30, "country": "Spain"},
            },
            "question_meta": {
                "original": "How does culture affect negotiation styles?",
                "sensitive_topics": ["Spain", "Mexico"],
                "relevant_cultures": ["Spain", "Mexico", "Argentina"],
            },
            "response_state": {},
        }
    )

    # Simulated activate_set (router output)
    state["activate_set"] = [
        (
            "Spain",
            0.9,
            "In Spain, negotiation often involves building personal relationships first.",
        ),
        (
            "Mexico",
            0.8,
            "Mexican negotiation styles emphasize courtesy and long-term trust.",
        ),
        (
            "Argentina",
            0.7,
            "Argentinian negotiators are known for strategic flexibility and patience.",
        ),
    ]

    output = compose_final_response(state)

    print("\n✅ Composed Final State:")
    for k, v in output.items():
        print(f"{k}: {v}")

    # Print final response word count
    final_text = output["response_state"]["final"]
    print("\n🔢 Final Response Word Count:", len(final_text.split()))
