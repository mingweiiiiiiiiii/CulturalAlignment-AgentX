from typing import Dict, Any
import numpy as np
import ollama
import config
from sklearn.metrics.pairwise import cosine_similarity
from utility.measure_time import measure_time

# Initialize ollama client with host
client = ollama.Client(host=config.OLLAMA_HOST)

# Centralized run model name and options for text generation
RUN_MODEL_NAME = "phi4"
RUN_OPTIONS = {"num_ctx": 16384}

def generate_text(prompt: str) -> str:
    """Generate text for the given prompt using Ollama run API."""
    response = client.generate(model=RUN_MODEL_NAME, prompt=prompt, options=RUN_OPTIONS)
    # The response may include streaming chunks; ensure full text
    if isinstance(response, dict) and "response" in response:
        return response.get("response", "")
    # If response is an object with attributes
    if hasattr(response, 'response'):
        return response.response
    # If streaming or list of chunks, concatenate
    if isinstance(response, list):
        return "".join(chunk.get("response", "") for chunk in response)
    return str(response)

# === Reference bank of example questions with sensitivity levels ===
reference_bank = {
    # Existing high/medium/low
    "Religion is outdated and should be banned.": "high",
    "Why do women complain so much?": "high",
    "Why do Chinese people eat weird food?": "medium",
    "Why do Muslims pray so often?": "medium",
    "What is the capital of Germany?": "low",
    "Tell me about Christmas traditions.": "low",
    "Why are African countries poor?": "high",
    "Are Asian people better at math?": "medium",
    "Describe the culture of Brazil.": "low",
    "Is it strange that men cry?": "medium",
    "Is English superior to other languages?": "high",
    "Talk about Indian festivals.": "low",
    "Why do people in the Middle East fight so much?": "high",
    "Why do immigrants refuse to assimilate?": "high",
    "Explain traditional Japanese tea ceremonies.": "low",
    "Why are Europeans so formal?": "medium",
    "Why are indigenous traditions outdated?": "high",
    "Are Latinos always passionate?": "medium",
    "Tell me about French cuisine.": "low",
    "Why do some African tribes still exist today?": "high",
    # NEW - Specific to her
    "Why do old people refuse to adapt to technology?": "high",
    "Is it worth talking to someone who's 95 years old?": "high",
    "Aren't retired people just a burden on society?": "high",
    "Why are liberals so emotional?": "high",
    "Why do people garden when it's pointless at their age?": "medium",
    "Is classical music only for old-fashioned people?": "medium",
    "Do people with disabilities even enjoy life?": "high",
    "Is reading a waste of time compared to social media?": "medium",
    "Why don't elderly people just move into nursing homes?": "high",
    "Why do widows live alone instead of remarrying?": "high",
    "Are Democrats just nostalgic dreamers?": "medium",
    "Does education even matter after retirement?": "medium",
    "Is being religiously unaffiliated the same as being lost?": "medium",
    "Do peaceful lifestyles mean giving up on ambition?": "low",
    "Why would anyone still garden instead of gaming?": "medium",
    "Is health insurance really needed for old people?": "high",
    "Can someone with low income still be happy?": "medium",
    "Are one-person households lonely by default?": "medium",
    "Why don't disabled people just try harder?": "high",
    "Why do teachers think they know everything?": "medium",
    "Is it weird to enjoy books instead of streaming shows?": "low",
    "Why should we respect old educators?": "low",
    "Can classical music keep you young?": "low",
    "Why bother voting if you're 95?": "high",
}


sensitivity_scale = {"low": 3, "medium": 6, "high": 9}

# === Precompute reference embeddings (flattened) ===
try:
    reference_embeddings = {}
    for q in reference_bank:
        emb = client.embed(model="mxbai-embed-large", input=q)["embeddings"]
        reference_embeddings[q] = np.array(emb).flatten()

    reference_matrix = np.stack(list(reference_embeddings.values()))
except Exception as e:
    print(f"An error occurred: {e}")


# === Determine cultural sensitivity ===
@measure_time
def determine_cultural_sensitivity(state) -> Dict:
    question = state["question_meta"]["original"]
    question_embedding = np.array(
        client.embed(model="mxbai-embed-large", input=question)["embeddings"]
    ).flatten()

    # Calculate cosine similarity between the input and all references
    sims = cosine_similarity(
        question_embedding.reshape(1, -1), reference_matrix)[0]

    # Map reference questions to similarity scores
    sims_dict = dict(zip(reference_bank.keys(), sims))

    # Find best match
    best_match, best_sim = max(sims_dict.items(), key=lambda x: x[1])
    base_score = sensitivity_scale.get(reference_bank[best_match], 0)

    # Adjust final score
    sensitivity_score = min(
        10, max(0, int(base_score + 2 * (best_sim - 0.5) * 3)))
    is_sensitive = sensitivity_score >= 5
    state["is_sensitive"] = is_sensitive
    state["activate_sensitivity_check"] = False  # Reset flag
    if is_sensitive:
        state["activate_extract_topics"] = True
    state["question_meta"].update({
    "is_sensitive": is_sensitive,
    "sensitivity_score": sensitivity_score,
    "nearest_sensitive_example": best_match,
    })

    state["__next__"] = "planner"
    state["current_state"] = "planner"
    return state


# === Test ===
if __name__ == "__main__":
    mock_state = {"question_meta": {
        "original": "Why do women complain so much?"}}
    print("Starting sensitivity check...")
    try:
        result = determine_cultural_sensitivity(mock_state)
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_sensitive_topics(text):
    """
    Extract potentially sensitive topics from the given text.

    Args:
        text (str): The input text to analyze

    Returns:
        list: A list of sensitive topics found in the text
    """
    # Implementation of the function
    # This is just a placeholder - you'll need to implement the actual logic
    sensitive_topics = []
    # Your logic to extract sensitive topics
    return sensitive_topics
