# main.py
# Orchestrates the culturally-aware multi-agent dialogue pipeline using LangGraph.
# This script serves as the entry point for the application, managing user profiles,
# embedding parameters, and evaluating responses from cultural experts.


import json
import pprint
from mylanggraph.graph import create_cultural_graph
from utility.inputData import PersonaSampler


def main():
    # Step 1: Input (User Profile + Questionnaire)
    sampler = PersonaSampler()
    profiles = sampler.sample_profiles(n=1)
    question, options = sampler.sample_question()
    prompt = sampler.build_prompt(profiles[0], question, options)

    print("\n=== Persona ===")
    print(json.dumps(profiles, indent=2))
    print("\n=== Question ===")
    print(question)
    print("\n=== Prompt ===")
    print(prompt)

    # Step 4: Create the cultural graph
    initial_state = {
        "user_profile": profiles,
        "question_meta": {"original": question, "options": options},
    }
    my_cultural_graph_workflow = create_cultural_graph()


    result = my_cultural_graph_workflow.invoke(
        initial_state,
        config={
            "recursion_limit": 200,  # Your existing config
            "configurable": {  # Add configurable block
                "thread_id": "1",  # Provide a thread ID
                # Optionally provide checkpoint_ns or checkpoint_id
                # "checkpoint_ns": "your_namespace",
                # "checkpoint_id": "your_checkpoint_id"
            }
        }
    )
    print("Printing ")
    pprint.pprint(result["response_state"]["final"])



   
    # # Get response from debate agent
    # response = create_cultural_graph(state=initial_state)


if __name__ == "__main__":
    main()
