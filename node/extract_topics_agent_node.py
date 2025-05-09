
from google import genai
from typing import Dict
from llmagentsetting import llm_clients
from utility.measure_time import measure_time

@measure_time
def extract_sensitive_topics(state) -> Dict:
    question = state["question_meta"]["original"]

    # Concatenate the prompt with the question and pass it to the GeminiClient
    prompt = f"""Identify potentially insensitive language or generalizations in the input.
    Focus on the core subject matter that could potentially involve:
    - stereotypes 
    - biases
    - disrespectful language
    - misunderstandings related to cultural practices, traditions, or specific groups of people.

    Output can contains one to many topics, separated by /

    Examples:
        Input: "I saw a group of people wearing those funny-looking hats at the temple today. What's the deal with that?"
        Topic: Religious Attire/Cultural Practices
        Input: "Why do Chinese always eat Stinky Tofu? It smells so strong."
        Topic: Stereotyping/Food Prejudice
        Input: "Is it true that in Japan, women aren't allowed to work after married?"
        Topic: Gender Roles/Cultural Misinformation/Generalization
        Input: "I don't understand why Asians make such a big deal about Lunar New Year. It seems so unnecessary."
        Topic: Cultural Value Systems/Dismissal of Tradition
        Input: "They all look the same to me. How do you even tell them apart?"
        Topic: Racial Bias/Microaggression

    Input:
    ```
    {question}
    ```     
    """

    client = llm_clients.LambdaAPIClient(state=state)
    topics = client.get_completion(prompt)

    # ✅ Proper in-place state updates
    state["question_meta"]["sensitive_topics"] = topics
    state["activate_extract_topics"] = False
    state["topics_extracted"] = True
    state["activate_router"] = True
    state["__next__"] = "planner"
    state["current_state"] = "planner"

    return state


if __name__ == "__main__":
    state = {
        "question_meta": {
            "original": "Why do different religions have different dietary restrictions and marriage customs?"
        }
    }

    result = extract_sensitive_topics(state)
    topics = result["question_meta"]["sensitive_topics"]
    print("Extracted Topics:", topics)
