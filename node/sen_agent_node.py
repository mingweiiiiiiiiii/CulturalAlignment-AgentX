import numpy as np
from typing import Dict
import ollama
from sklearn.metrics.pairwise import cosine_similarity

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
        emb = ollama.embed(model="mxbai-embed-large", input=q)["embeddings"]
        reference_embeddings[q] = np.array(emb).flatten()   

    reference_matrix = np.stack(list(reference_embeddings.values()))
except Exception as e:
    print(f"An error occurred: {e}")


# === Determine cultural sensitivity ===
def determine_cultural_sensitivity(state) -> Dict:
    question = state["question_meta"]["original"]
    question_embedding = np.array(
        ollama.embed(model="mxbai-embed-large", input=question)["embeddings"]
    ).flatten()

    # Calculate cosine similarity between the input and all references
    sims = cosine_similarity(question_embedding.reshape(1, -1), reference_matrix)[0]

    # Map reference questions to similarity scores
    sims_dict = dict(zip(reference_bank.keys(), sims))

    # Find best match
    best_match, best_sim = max(sims_dict.items(), key=lambda x: x[1])
    base_score = sensitivity_scale.get(reference_bank[best_match], 0)

    # Adjust final score
    sensitivity_score = min(10, max(0, int(base_score + 2 * (best_sim - 0.5) * 3)))

    return {
        "question_meta": {
            **state["question_meta"],
            "is_sensitive": sensitivity_score >= 5,
            "sensitivity_score": sensitivity_score,
            "nearest_sensitive_example": best_match,
        },
        "current_state": "sensitivity_check",
    }


# === Test ===
if __name__ == "__main__":
    mock_state = {"question_meta": {"original": "Why do women complain so much?"}}
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
