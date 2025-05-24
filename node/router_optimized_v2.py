"""
Optimized router v2 with full 20 culture pool and smart expert selection.
Limits to top 5 experts maximum.
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from node.cultural_expert_node_smart import CulturalExpertManager
from node.embed_utils import embed_persona
import ollama
import config
from utility.measure_time import measure_time
from functools import lru_cache

def embed_text(text: str) -> np.ndarray:
    """Embed text using ollama."""
    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        response = client.embed(model="mxbai-embed-large", input=text)
        return np.array(response["embeddings"]).flatten()
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

# Configuration
MAX_EXPERTS = 5  # Maximum number of experts to select
MIN_EXPERTS = 3  # Minimum number of experts (for diversity)
RELEVANCE_THRESHOLD = 5.0  # Minimum score for full response

@measure_time
def route_to_cultures_smart(state: Dict) -> Dict:
    """
    Smart router that:
    1. Selects top K (max 5) cultures from pool of 20
    2. Only generates full responses for culturally relevant questions
    """
    question_meta = state.get("question_meta", {})
    user_profile = state.get("user_profile", {})
    
    # Get question and topics
    question = question_meta.get("original", "")
    sensitive_topics = question_meta.get("sensitive_topics", [])
    relevant_cultures = question_meta.get("relevant_cultures", [])
    
    # Initialize manager with full 20 culture pool
    manager = CulturalExpertManager(state=state)
    expert_list, all_embeddings = manager.get_all_persona_embeddings()
    
    print(f"Selecting from pool of {len(expert_list)} cultures")
    
    # Compute embeddings for scoring
    if sensitive_topics:
        topic_embeddings = []
        for topic in sensitive_topics:
            emb = embed_text(f"Cultural perspectives on {topic}")
            if emb is not None:
                topic_embeddings.append(emb)
        if topic_embeddings:
            topic_centroid = np.mean(topic_embeddings, axis=0)
        else:
            topic_centroid = embed_text(question)
    else:
        topic_centroid = embed_text(question)
    
    # User embedding
    user_embedding = embed_persona(user_profile) if user_profile else None
    
    # Score all cultures
    scores = []
    for i, culture in enumerate(expert_list):
        # Base score from embedding similarity
        culture_emb = all_embeddings[i]
        
        # Topic similarity (60% weight)
        if topic_centroid is not None:
            topic_sim = cosine_similarity(culture_emb, topic_centroid)
        else:
            topic_sim = 0.5
        
        # User similarity (40% weight)
        if user_embedding is not None:
            user_sim = cosine_similarity(culture_emb, user_embedding)
        else:
            user_sim = 0.5
        
        # Combined score
        score = 0.6 * topic_sim + 0.4 * user_sim
        
        # Boost if culture is explicitly mentioned as relevant
        if culture in relevant_cultures:
            score += 0.2
        
        # Boost if culture matches user location
        if culture == user_profile.get("location"):
            score += 0.15
        
        scores.append((culture, min(score, 1.0)))
    
    # Sort by score and select top K
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Determine how many to select
    num_to_select = min(MAX_EXPERTS, max(MIN_EXPERTS, len([s for s in scores if s[1] > 0.5])))
    selected_cultures = [culture for culture, score in scores[:num_to_select]]
    
    print(f"Selected {len(selected_cultures)} cultures: {', '.join(selected_cultures)}")
    
    # Get smart responses (only full responses for relevant cultures)
    expert_responses = manager.get_smart_expert_responses(
        question, 
        selected_cultures,
        relevance_threshold=RELEVANCE_THRESHOLD
    )
    
    # Count how many gave full vs brief responses
    full_responses = sum(1 for r in expert_responses.values() if r['response_type'] == 'full')
    brief_responses = len(expert_responses) - full_responses
    
    print(f"Response breakdown: {full_responses} full, {brief_responses} brief")
    
    # Update state with results
    state["expert_responses"] = expert_responses
    state["selected_cultures"] = selected_cultures
    state["culture_scores"] = dict(scores[:num_to_select])
    
    # Route to composition
    state["activate_compose"] = True
    state["steps"].append(f"Selected {len(selected_cultures)} experts from pool of 20")
    state["steps"].append(f"Generated {full_responses} full and {brief_responses} brief responses")
    
    return state

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

# Pre-compute fallback centroids for speed
@lru_cache(maxsize=1)
def get_precomputed_centroids(n_clusters: int = 5) -> np.ndarray:
    """Pre-compute cluster centroids for fallback mechanism."""
    manager = CulturalExpertManager()
    _, embeddings = manager.get_all_persona_embeddings()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    print(f"Pre-computed {n_clusters} centroids from {len(embeddings)} cultures")
    return kmeans.cluster_centers_

if __name__ == "__main__":
    # Test the smart router
    test_state = {
        "question_meta": {
            "original": "What role should religion play in government?",
            "sensitive_topics": ["religion", "politics", "governance"],
            "relevant_cultures": ["United States", "Egypt", "Turkey"]
        },
        "user_profile": {
            "location": "Turkey",
            "cultural_background": "Middle Eastern",
            "age": 40
        },
        "steps": []
    }
    
    print("Testing smart router with full culture pool...")
    result = route_to_cultures_smart(test_state)
    
    print(f"\nSelected cultures: {result['selected_cultures']}")
    print(f"\nExpert responses:")
    for culture, info in result['expert_responses'].items():
        print(f"\n{culture}:")
        print(f"  Relevance: {info['relevance_score']}/10")
        print(f"  Type: {info['response_type']}")
        print(f"  Response length: {len(info['response'])} chars")