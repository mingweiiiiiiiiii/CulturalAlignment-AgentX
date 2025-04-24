# main.py
# Orchestrates the culturally-aware multi-agent dialogue pipeline using LangGraph.
# This script serves as the entry point for the application, managing user profiles,
# embedding parameters, and evaluating responses from cultural experts.


import json
import random
from inputData import PersonaSampler
from WorldValueSurveyProcess import embed_persona
from db import insert_embedding
from graph import create_cultural_graph
from evaluation import evaluate_all_metrics, save_evaluation_chart, EvaluationLosses
from llm_manager import LLMManager
from llm_manager import LLMManager

def main():
    # Step 1: Input (User Profile + Questionnaire)
    sampler = PersonaSampler()
    profile = sampler.sample_profiles(n=1)[0]
    question, options = sampler.sample_question()
    prompt = sampler.build_prompt(profile, question, options)

    print("\n=== Persona ===")
    print(json.dumps(profile, indent=2))
    print("\n=== Question ===")
    print(question)
    print("\n=== Prompt ===")
    print(prompt)

    # Step 2: Embed parameters
    embedding = embed_persona(profile)

    # Step 3: Store vector in database
    insert_embedding(embedding)  # Store the user embedding
    # Retrieve similar embeddings if needed
    similar_embeddings = search_similar(embedding)

    # Step 4: Create the cultural graph
    graph = create_cultural_graph()

    # Step 5: Prepare initial state
    initial_state = {
        "user_profile": profile,
        "question_meta": {
            "original": question,
            "options": options
        },
        "user_embedding": embedding
    }

    # Step 6: Run the graph
    final_state = graph.run(initial_state)

    # Step 7: Evaluate metrics
    responses = final_state.get("response_state", {}).get("expert_responses", [])
    # Initialize evaluation metrics
    lambdas = [1.0] * 7  # Example weights for each metric
    label_map = {"safe": 0, "topic": 1}  # Example label mapping
    evaluator = EvaluationLosses(lambdas, label_map)
    
    # Calculate metrics
    response_pack = {
        'response': final_state['response'],
        'topic_responses': responses,
        'topics': [question],  # Example topics
        'cultural_ref': "Cultural Reference",  # Placeholder
        'style': "Target Style",  # Placeholder
        'same_culture_responses': responses,  # Example
        'responseA': responses[0],  # Example
        'responseB': responses[1],  # Example
        'predictions': [],  # Placeholder
        'labels': [],  # Placeholder
        'masks': []  # Placeholder
    }
    
    total_loss = evaluator.L_total(response_pack)
    print(f"Total Loss: {total_loss}")

    # Save evaluation chart
    save_evaluation_chart([total_loss])  # Placeholder for actual data

if __name__ == "__main__":
    main()
