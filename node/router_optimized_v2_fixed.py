"""
Smart cultural router with full 20 culture pool and top K selection.
Fixed to populate relevant_cultures for alignment score calculation.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from sklearn.cluster import KMeans
import time

from node.embed_utils import embed_persona, get_embeddings
from node.cultural_expert_node_smart import CulturalExpertManager
from utility.measure_time import measure_time

# Configuration for smart routing
MAX_EXPERTS = 5  # Maximum number of experts to select
MIN_EXPERTS = 3  # Minimum number of experts (for diversity)
RELEVANCE_THRESHOLD = 5.0  # Minimum score for full response

@measure_time
def route_to_cultures_smart(state: Dict) -> Dict:
    """
    Smart routing to top K cultures from full pool of 20.
    Only generates full responses for culturally relevant questions.
    
    FIXED: Now populates question_meta["relevant_cultures"] for alignment scoring.
    """
    question_meta = state.get("question_meta", {})
    question = question_meta.get("original", "")
    sensitive_topics = question_meta.get("sensitive_topics", [])
    user_profile = state.get("user_profile", {})
    is_sensitive = question_meta.get("is_sensitive", False)
    
    print(f"\n=== Smart Cultural Routing ===")
    print(f"Question sensitive: {is_sensitive}")
    print(f"Sensitive topics: {sensitive_topics}")
    
    # Initialize manager
    manager = CulturalExpertManager()
    
    # Generate expert instances if not already done
    if not manager.expert_instances:
        manager.generate_expert_instances()
    
    # Get embeddings for all cultures
    expert_list, all_embeddings = manager.get_all_persona_embeddings()
    
    # Prepare for routing
    relevant_cultures = question_meta.get("relevant_cultures", [])
    
    # Topic embedding (from sensitive topics or question)
    if sensitive_topics:
        topic_embeddings = [get_embeddings(topic) for topic in sensitive_topics[:3]]
        if topic_embeddings:
            topic_centroid = np.mean(np.stack(topic_embeddings), axis=0)
        else:
            topic_centroid = np.array(get_embeddings(question))
    else:
        topic_centroid = np.array(get_embeddings(question))
    
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
    
    # FIXED: Populate relevant_cultures for alignment score calculation
    # These are the cultures most relevant to this user based on their profile
    state["question_meta"]["relevant_cultures"] = selected_cultures
    
    # Route to composition
    state["activate_compose"] = True
    
    # Handle steps field gracefully
    if "steps" not in state:
        state["steps"] = []
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
            "original": "Should women prioritize career advancement or family responsibilities?",
            "sensitive_topics": ["gender roles", "family", "career"],
            "is_sensitive": True,
            "sensitivity_score": 8
        },
        "user_profile": {
            "age": 35,
            "sex": "Female",
            "place_of_birth": "Japan",
            "ethnicity": "Japanese",
            "education": "Bachelor's degree"
        },
        "expert_responses": {},
        "steps": []
    }
    
    result = route_to_cultures_smart(test_state)
    
    print(f"\nSelected cultures: {result['selected_cultures']}")
    print(f"\nRelevant cultures saved: {result['question_meta']['relevant_cultures']}")
    print(f"\nExpert responses:")
    for culture, info in result['expert_responses'].items():
        print(f"\n{culture}:")
        print(f"  Relevance: {info['relevance_score']}/10")
        print(f"  Type: {info['response_type']}")
        print(f"  Response length: {len(info['response'])} chars")