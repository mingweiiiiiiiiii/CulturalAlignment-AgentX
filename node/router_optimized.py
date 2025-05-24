import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from node.embed_utils import embed_topics
from node.cultural_expert_node import CulturalExpertManager
from utility.measure_time import measure_time
from utility.cache_manager import get_cache_manager

# Global variable to store pre-computed centroids
_precomputed_centroids = None

def precompute_centroids(state: Dict = None) -> np.ndarray:
    """Pre-compute KMeans centroids for fallback routing"""
    global _precomputed_centroids
    
    if _precomputed_centroids is not None:
        return _precomputed_centroids
    
    print("Pre-computing cultural centroids...")
    # Create a minimal state if none provided
    if state is None:
        state = {"user_profile": {}}
    manager = CulturalExpertManager(state=state)
    expert_list, all_embeddings = manager.get_all_persona_embeddings()
    
    # Compute centroids using KMeans
    n_clusters = min(3, len(expert_list))  # Default to 3 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(all_embeddings)
    
    _precomputed_centroids = kmeans.cluster_centers_
    print(f"Pre-computed {n_clusters} centroids")
    
    return _precomputed_centroids

@measure_time
def route_to_cultures_optimized(state: Dict) -> Dict:
    """
    Optimized router that uses pre-computed centroids and caching.
    """
    question_meta = state.get("question_meta", {})
    user_profile = state.get("user_profile", {})
    
    # Get sensitive topics and embeddings
    sensitive_topics = question_meta.get("sensitive_topics", [])
    relevant_cultures = question_meta.get("relevant_cultures", [])
    
    # Initialize manager and get embeddings
    manager = CulturalExpertManager(state=state)
    expert_list, all_embeddings = manager.get_all_persona_embeddings()
    
    # Get cache manager
    cache = get_cache_manager()
    
    # Compute topic embeddings with caching
    if sensitive_topics:
        # Ensure sensitive_topics is a list
        if isinstance(sensitive_topics, str):
            sensitive_topics = [sensitive_topics]
        
        print(f"Computing embeddings for {len(sensitive_topics)} sensitive topics...")
        topic_embeddings = []
        
        for topic in sensitive_topics:
            if not topic or not isinstance(topic, str):
                continue
                
            # Check cache first
            cached_embedding = cache.get_embedding(topic)
            if cached_embedding is not None:
                topic_embeddings.append(cached_embedding)
            else:
                # Compute and cache
                embedding = embed_topics({"topics": [topic]})[0]
                cache.put_embedding(topic, embedding)
                topic_embeddings.append(embedding)
        
        if topic_embeddings:
            topic_embeddings = np.array(topic_embeddings)
            print(f"Topic embeddings shape: {topic_embeddings.shape}")
        else:
            # No valid topics, use centroids
            topic_embeddings = precompute_centroids(state)
            print(f"No valid topics, using {len(topic_embeddings)} pre-computed centroids")
    else:
        # Use pre-computed centroids for fallback
        topic_embeddings = precompute_centroids(state)
        print(f"Using {len(topic_embeddings)} pre-computed centroids")
    
    # Batch compute similarities
    similarities = cosine_similarity(topic_embeddings, all_embeddings)
    
    # Aggregate similarities
    if len(topic_embeddings) > 1:
        # Average similarity across all topics
        avg_similarities = similarities.mean(axis=0)
    else:
        avg_similarities = similarities[0]
    
    # Create activate_set with culture weights
    activate_set = []
    
    # Add explicitly mentioned cultures with boost
    mentioned_boost = 0.2
    for i, culture in enumerate(expert_list):
        base_score = avg_similarities[i]
        
        # Boost if culture is mentioned in relevant_cultures
        if culture in relevant_cultures:
            base_score += mentioned_boost
        
        activate_set.append({
            "culture": culture,
            "weight": float(base_score),
            "embedding_idx": i
        })
    
    # Sort by weight
    activate_set = sorted(activate_set, key=lambda x: x["weight"], reverse=True)
    
    # Apply minimum threshold
    min_threshold = 0.1
    activate_set = [entry for entry in activate_set if entry["weight"] >= min_threshold]
    
    # Ensure at least 3 cultures are selected
    if len(activate_set) < 3:
        # Add top remaining cultures
        remaining = [entry for entry in sorted(activate_set, key=lambda x: x["weight"], reverse=True)]
        activate_set = remaining[:3]
    
    print(f"Selected {len(activate_set)} cultures for consultation:")
    for entry in activate_set[:5]:  # Show top 5
        print(f"  - {entry['culture']}: {entry['weight']:.3f}")
    
    # Update state
    state["activate_set"] = activate_set
    state["activate_router"] = False
    
    return {
        "activate_set": activate_set
    }