from typing import Dict, List, Tuple

import numpy as np
import torch

from culturalRouting import (expert_classes, get_text_embedding,
                             route_to_cultures)
from llm_provider import create_llm_provider

# Create a single LLM provider instance to be reused
_llm_provider = create_llm_provider("mock")


def llm_call(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Calls the LLM provider to generate a response to the given prompt.

    Args:
        prompt: The text prompt to send to the model
        model: Model identifier (used only for provider-specific configuration)

    Returns:
        Generated text response from the LLM
    """
    # Note: The model parameter is kept for compatibility but not used
    # since model selection happens when creating the provider
    return _llm_provider.generate(prompt)


def initiate_debate(state: Dict, num_rounds: int = 3) -> Dict:
    """
    Initiates a debate among the cultures selected in the routing phase.

    Args:
        state: The state after cultural routing
        num_rounds: Number of debate rounds to conduct

    Returns:
        Updated state with debate results
    """
    # Get cultures and their prompts from cultural routing if not already present
    if "expert_weights_and_prompts" not in state:
        state = route_to_cultures(state)

    expert_weights_and_prompts = state["expert_weights_and_prompts"]

    # Extract the original question
    original_question = state["question_meta"]["original"]
    sensitive_topics = state["question_meta"].get("sensitive_topics", [])

    # Initialize debate history
    debate_history = []

    # First round: Initial responses from each culture
    first_round_responses = []
    for culture, weight, prompt in expert_weights_and_prompts:
        # Enhance the prompt to get an initial response
        initial_prompt = f"""
        {prompt}
        
        You are providing the initial perspective for this question based on your cultural viewpoint.
        Consider how this topic is viewed in your culture and provide key insights.
        Keep your response concise (100-150 words) and focused on your cultural perspective.
        
        Question: {original_question}
        Topics: {', '.join(sensitive_topics) if sensitive_topics else 'general'}
        """

        response = llm_call(initial_prompt)
        first_round_responses.append({
            "culture": culture,
            "weight": weight,
            "response": response
        })

    # Add first round to debate history
    debate_history.append({
        "round": 1,
        "responses": first_round_responses
    })

    # Subsequent rounds: Cultures respond to each other
    for round_num in range(2, num_rounds + 1):
        round_responses = []

        # Format previous round responses as context
        previous_round = debate_history[-1]["responses"]
        debate_context = "\n\n".join([
            f"{resp['culture']}: {resp['response']}"
            for resp in previous_round
        ])

        for culture, weight, _ in expert_weights_and_prompts:
            # Create prompt that includes previous responses
            debate_prompt = f"""
            Original Question: {original_question}
            Topics: {', '.join(sensitive_topics) if sensitive_topics else 'general'}
            
            Previous Round Perspectives:
            {debate_context}
            
            You represent the {culture} perspective. Based on the discussion so far:
            1. Acknowledge valid points made by others
            2. Present additional insights from your cultural perspective
            3. Respectfully note where you might differ from other perspectives
            
            Keep your response concise (100-150 words) and constructive.
            """

            response = llm_call(debate_prompt)
            round_responses.append({
                "culture": culture,
                "weight": weight,
                "response": response
            })

        # Add this round to debate history
        debate_history.append({
            "round": round_num,
            "responses": round_responses
        })

    # Final synthesis of perspectives weighted by culture weights
    cultures = [culture for culture, _, _ in expert_weights_and_prompts]
    weights = [weight for _, weight, _ in expert_weights_and_prompts]

    # First construct the debate transcript separately
    debate_transcript = ""
    for round_data in debate_history:
        debate_transcript += f"\n--- ROUND {round_data['round']} ---\n"
        for resp in round_data['responses']:
            debate_transcript += f"{resp['culture']}: {resp['response']}\n"

    synthesis_prompt = f"""
    Original Question: {original_question}
    Topics: {', '.join(sensitive_topics) if sensitive_topics else 'general'}
    
    You have moderated a debate between representatives of these cultures: {', '.join(cultures)}.
    Consider that their relative importance weights are: {', '.join([f"{c}({w:.2f})" for c, w in zip(cultures, weights)])}.
    
    The full debate transcript is below:
    
    {debate_transcript}
    
    Please synthesize the key insights from this cultural debate into a balanced response that:
    1. Acknowledges the different cultural perspectives
    2. Highlights points of agreement and disagreement
    3. Provides a nuanced answer to the original question
    4. Gives more weight to perspectives with higher importance weights
    """

    synthesis = llm_call(synthesis_prompt)

    # Update state with debate results
    return {
        **state,
        "current_state": "debate_complete",
        "debate_history": debate_history,
        "cultural_synthesis": synthesis
    }


def generate_debate_response(state: Dict) -> str:
    """
    Generates a formatted response from the debate results.

    Args:
        state: The state object containing debate history and synthesis

    Returns:
        A formatted string for presenting to the user
    """
    if state.get("current_state") != "debate_complete":
        return "No debate has been conducted yet."

    synthesis = state.get("cultural_synthesis", "")
    debate_history = state.get("debate_history", [])

    # Format the full response
    cultures = [resp["culture"] for resp in debate_history[0]["responses"]]

    response = f"# Cultural Perspectives on Your Question\n\n"
    response += f"I consulted representatives from {', '.join(cultures)} to provide a nuanced answer.\n\n"

    # Add synthesis
    response += f"## Synthesis\n\n{synthesis}\n\n"

    # Optionally add debate highlights
    response += f"## Highlights from the Cultural Exchange\n\n"

    for round_data in debate_history:
        round_num = round_data["round"]
        response += f"### Round {round_num}\n\n"

        for resp in round_data["responses"]:
            culture = resp["culture"]
            content = resp["response"]
            response += f"**{culture}**: {content}\n\n"

    return response


# Example usage
if __name__ == "__main__":
    # Example state similar to the one in culturalRouting.py
    example_state = {
        "question_meta": {
            "original": "How should we address gender roles in different societies?",
            "sensitive_topics": ["gender", "roles"]
        },
        "user_profile": {
            "id": "user_123",
            "demographics": {"age": 30, "region": "Europe"},
            "preferences": {"tone": "academic"}
        }
    }

    # First get the routing
    routed_state = route_to_cultures(example_state)

    # Then run the debate
    debate_state = initiate_debate(routed_state, num_rounds=2)

    # Generate the final response
    response = generate_debate_response(debate_state)
    print(response[:500] + "...")  # Print beginning of response
