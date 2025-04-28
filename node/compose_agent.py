from typing import Dict, List, Tuple, Any
import google.generativeai as genai
# Define ExpertResponse type
ExpertResponse = Dict[str, str]


# ===============================
# 🎯 Compose Final Response
# ===============================
class GraphState(dict):
    """A simple extension of dict for holding graph states."""



def compose_final_response(
    state,
    activate_set: List[Tuple[str, float, str]],  # (culture, weight, prompt_j)
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    Compose a final response based on multiple cultural expert inputs,
    aiming for a concise culturally-informed answer within 200 words.
    """

    user_profile = state.get("user_profile", {})
    question_meta = state.get("question_meta", {})
    preferences = user_profile.get("preferences", {})
    demographics = user_profile.get("demographics", {})

    # Step 1: Select top-N cultures by weight
    top_cultures = sorted(activate_set, key=lambda x: x[1], reverse=True)[:top_n]

    expert_responses: List[ExpertResponse] = [
        {"culture": culture, "response": response}
        for culture, _, response in top_cultures
    ]

    # Step 2: Create LLM prompt
    question = question_meta.get("original", "")
    sensitive_topics = question_meta.get("sensitive_topics", [])
    relevant_cultures = question_meta.get("relevant_cultures", [])

    prompt_parts = [
        "You are a culturally-aware assistant tasked with composing a final response that is sensitive to the user's background and preferences.",
        "Please synthesize the expert responses and produce a final answer that is coherent, respectful, and **under 200 words**.",
        f"Question: {question}",
        "\nUser Profile:",
        f"- Demographics: {demographics}",
        f"- Preferences: {preferences}",
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
    GoogleStudio_API_KEY = "AIzaSyAlMLq2h1YHKJgOm6hds2aHz_iWrByXacM"
    # Create model
    model = genai.Client(api_key=GoogleStudio_API_KEY)

    final_response = model.models.generate_content(
        model="gemini-2.0-flash", contents=prompt_parts
    ).text

    # Step 4: Optional soft enforcement (post-process if needed)
    words = final_response.split()
    if len(words) > 200:
        final_response = " ".join(words[:200]) + "..."

    # Step 5: Update response state
    response_state = state.get("response_state", {})
    response_state.update(
        {"expert_responses": expert_responses, "final": final_response}
    )

    return {"response_state": response_state, "current_state": "compose"}


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
    activate_set = [
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

    output = compose_final_response(state, activate_set, top_n=2)

    print("\n✅ Composed Final State:")
    for k, v in output.items():
        print(f"{k}: {v}")

    # Print final response word count
    final_text = output["response_state"]["final"]
    print("\n🔢 Final Response Word Count:", len(final_text.split()))
