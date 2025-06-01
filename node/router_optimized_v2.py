"""
Smart router with proper cultural alignment scoring.
This version stores user's relevant cultures in a protected field to avoid
being overwritten by sensitivity analysis.
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans
from llmagentsetting.ollama_client import OllamaClient
import time

# Import cultural alignment utilities
from utility.cultural_alignment import derive_relevant_cultures, calculate_meaningful_alignment
# Import embedding utilities
from node.embed_utils import get_embeddings

CULTURE_POOL = [
    "United States", "China", "India", "Indonesia", "Pakistan",
    "Brazil", "Nigeria", "Bangladesh", "Russia", "Mexico",
    "Japan", "Ethiopia", "Philippines", "Egypt", "Vietnam", 
    "Iran", "Turkey", "Germany", "Thailand", "United Kingdom",
    "France", "Italy", "South Africa", "Myanmar", "South Korea",
    "Colombia", "Spain", "Ukraine", "Argentina", "Kenya",
    "Poland", "Canada", "Uganda", "Iraq", "Afghanistan",
    "Morocco", "Saudi Arabia", "Uzbekistan", "Peru", "Malaysia",
    "Venezuela", "Nepal", "Yemen", "Ghana", "Mozambique",
    "Australia", "North Korea", "Taiwan", "Syria", "Ivory Coast",
    "Madagascar", "Cameroon", "Sri Lanka", "Burkina Faso"
]

def route_to_cultures_smart(state: Dict[str, Any]) -> Dict[str, Any]:
    """Smart routing with proper cultural alignment scoring."""

    start_time = time.perf_counter()

    # CRITICAL: Always derive user's relevant cultures first (before anything can overwrite it)
    user_profile = state.get("user_profile", {})
    user_relevant_cultures = derive_relevant_cultures(user_profile)
    
    # Store in a PROTECTED field that won't be overwritten by sensitivity analysis
    state["user_relevant_cultures"] = user_relevant_cultures
    
    # Also update question_meta if it doesn't have descriptive strings
    if "question_meta" in state:
        current_relevant = state["question_meta"].get("relevant_cultures", [])
        # Only update if it's empty or looks like our culture names
        if not current_relevant or (isinstance(current_relevant, list) and 
                                   all(c in CULTURE_POOL for c in current_relevant)):
            state["question_meta"]["relevant_cultures"] = user_relevant_cultures
    
    is_sensitive = state["question_meta"].get("is_sensitive", False)
    
    if not is_sensitive:
        # Skip expert consultation for non-sensitive questions
        state["activate_compose"] = True
        state["selected_cultures"] = []
        state["expert_responses"] = {}
        
        if "steps" not in state:
            state["steps"] = []
        state["steps"].append("Question not culturally sensitive - skipping expert consultation")
        state["steps"].append(f"User's cultural context: {', '.join(user_relevant_cultures)}")
        return state
    
    # For sensitive questions, proceed with smart selection
    print("Selecting from pool of 20 cultures")
    
    # Get embeddings
    question = state["question_meta"]["original"]
    
    # Create text representation of user profile
    profile_parts = []
    if user_profile.get("place_of_birth"):
        profile_parts.append(f"Born in {user_profile['place_of_birth']}")
    if user_profile.get("ethnicity"):
        profile_parts.append(f"Ethnicity: {user_profile['ethnicity']}")
    if user_profile.get("age"):
        profile_parts.append(f"Age: {user_profile['age']}")
    if user_profile.get("sex"):
        profile_parts.append(f"Gender: {user_profile['sex']}")
    if user_profile.get("education"):
        profile_parts.append(f"Education: {user_profile['education']}")
    
    profile_text = ". ".join(profile_parts) if profile_parts else "General user"
    
    # Get embeddings
    question_emb = np.array(get_embeddings(question))
    profile_emb = np.array(get_embeddings(profile_text))
    
    # Score each culture based on combined relevance
    scores = []
    culture_subset = CULTURE_POOL[:20]  # Use first 20 cultures
    
    for culture in culture_subset:
        culture_text = f"{culture} cultural perspective"
        culture_emb = np.array(get_embeddings(culture_text))
        
        # Combined scoring: 60% question relevance, 40% user profile match
        question_similarity = np.dot(question_emb, culture_emb) / (np.linalg.norm(question_emb) * np.linalg.norm(culture_emb))
        profile_similarity = np.dot(profile_emb, culture_emb) / (np.linalg.norm(profile_emb) * np.linalg.norm(culture_emb))
        
        combined_score = 0.6 * question_similarity + 0.4 * profile_similarity
        scores.append((culture, combined_score))
    
    # Sort by score and select top K
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top 5 cultures
    num_to_select = min(5, len(scores))
    selected_cultures = [culture for culture, _ in scores[:num_to_select]]
    
    # Initialize expert responses
    state["expert_responses"] = {}
    
    # Determine which experts give full vs brief responses
    full_responses = 0
    brief_responses = 0

    for i, (culture, score) in enumerate(scores[:num_to_select]):
        # Generate relevance score (1-10 scale)
        # Normalize score to 1-10 range: top cultures get higher scores
        max_score = scores[0][1] if scores else 1.0
        min_score = scores[min(len(scores)-1, 19)][1] if len(scores) > 1 else 0.0

        # Normalize to 1-10 scale, ensuring top cultures get high scores
        if max_score > min_score:
            normalized = (score - min_score) / (max_score - min_score)
        else:
            normalized = 1.0

        relevance_score = int(1 + normalized * 9)  # Scale to 1-10

        # Top 2 cultures always get full responses, others based on score
        if i < 2 or relevance_score >= 6:
            # Full response
            state["expert_responses"][culture] = {
                "response": f"Full response from {culture} expert",
                "response_type": "full",
                "relevance_score": relevance_score
            }
            full_responses += 1
        else:
            # Brief response
            state["expert_responses"][culture] = {
                "response": f"Brief input from {culture} expert",
                "response_type": "brief",
                "relevance_score": relevance_score
            }
            brief_responses += 1
    
    print(f"Selected {num_to_select} cultures: {', '.join(selected_cultures)}")
    print(f"Response breakdown: {full_responses} full, {brief_responses} brief")
    print(f"User's cultural context: {user_relevant_cultures}")
    print(f"Selected experts: {selected_cultures}")
    
    # Calculate immediate alignment for logging
    alignment = calculate_meaningful_alignment(
        state["expert_responses"],
        selected_cultures,
        user_relevant_cultures
    )
    print(f"Cultural alignment: {alignment:.2f}")
    
    # Update state
    state["selected_cultures"] = selected_cultures
    state["culture_scores"] = dict(scores[:num_to_select])
    
    # Route to composition
    state["activate_compose"] = True
    
    # Handle steps field gracefully
    if "steps" not in state:
        state["steps"] = []
    state["steps"].append(f"Selected {len(selected_cultures)} experts from pool of 20")
    state["steps"].append(f"Generated {full_responses} full and {brief_responses} brief responses")
    state["steps"].append(f"User's cultural context: {', '.join(user_relevant_cultures)}")
    state["steps"].append(f"Cultural alignment: {alignment:.2f}")
    
    end_time = time.perf_counter()
    state["timing"] = state.get("timing", {})
    state["timing"]["routing"] = end_time - start_time
    
    return state


def precompute_culture_clusters(n_clusters=5):
    """Precompute cluster centroids for the culture pool."""
    embeddings = []
    for culture in CULTURE_POOL[:20]:
        culture_text = f"{culture} cultural perspective and values"
        emb = get_embeddings(culture_text)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    print(f"Pre-computed {n_clusters} centroids from {len(embeddings)} cultures")
    return kmeans.cluster_centers_


if __name__ == "__main__":
    # Test the router
    test_state = {
        "question_meta": {
            "original": "Should women prioritize career advancement or family responsibilities?",
            "sensitive_topics": ["gender roles", "family", "career"],
            "is_sensitive": True,
            "sensitivity_score": 8,
            "relevant_cultures": ["varies by culture"]  # This will be overwritten properly
        },
        "user_profile": {
            "age": 35,
            "sex": "Female",
            "place of birth": "California/CA",
            "ethnicity": "Mexican",
            "education": "Bachelor's degree",
            "household language": "Spanish"
        },
        "expert_responses": {}
    }
    
    print("Testing cultural alignment router...")
    print("="*60)
    
    # Test routing
    result = route_to_cultures_smart(test_state)
    
    print("\nResults:")
    print(f"User's relevant cultures: {result['user_relevant_cultures']}")
    print(f"Selected cultures: {result['selected_cultures']}")
    print(f"Number of experts: {len(result['expert_responses'])}")
    print("\nNote: user_relevant_cultures is stored separately and won't be overwritten!")