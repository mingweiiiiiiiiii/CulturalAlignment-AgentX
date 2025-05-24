"""
Demonstration of cultural alignment with full 20-culture pool.
Shows how the system selects top K most relevant experts.
"""
import sys
sys.path.append('/app')

from typing import Dict, List
import numpy as np
from datetime import datetime
import json

# Configuration
TOP_K_EXPERTS = 5  # Select top 5 experts from pool of 20

# Full culture pool
CULTURE_POOL = [
    "United States", "China", "India", "Japan", "Turkey", 
    "Vietnam", "Russia", "Brazil", "South Africa", "Germany",
    "France", "Italy", "Spain", "Mexico", "Egypt",
    "Kenya", "Nigeria", "Indonesia", "Philippines", "Thailand"
]

def demonstrate_expert_selection():
    """Demonstrate how experts are selected from the full pool."""
    
    print("=" * 80)
    print("CULTURAL EXPERT SELECTION DEMONSTRATION")
    print("=" * 80)
    print(f"Total Culture Pool: {len(CULTURE_POOL)} cultures")
    print(f"Selection Strategy: Top {TOP_K_EXPERTS} most relevant based on embedding similarity")
    print("-" * 80)
    
    # Test scenarios
    test_cases = [
        {
            "question": "What are your views on arranged marriages versus love marriages?",
            "user_profile": {
                "location": "India",
                "cultural_background": "South Asian",
                "age": 28
            },
            "expected_relevant": ["India", "Pakistan", "Bangladesh", "Nepal"],
            "sensitivity_score": 9
        },
        {
            "question": "Should companies prioritize profit or social responsibility?",
            "user_profile": {
                "location": "United States",
                "cultural_background": "Western",
                "age": 45
            },
            "expected_relevant": ["United States", "Germany", "France", "Italy"],
            "sensitivity_score": 6
        },
        {
            "question": "How important is respecting elders in society?",
            "user_profile": {
                "location": "Japan",
                "cultural_background": "East Asian",
                "age": 35
            },
            "expected_relevant": ["Japan", "China", "Vietnam", "Thailand"],
            "sensitivity_score": 7
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}")
        print(f"{'='*60}")
        print(f"Question: {test['question']}")
        print(f"User: {test['user_profile']['location']}, {test['user_profile']['cultural_background']}")
        print(f"Sensitivity Score: {test['sensitivity_score']}/10")
        
        # Simulate expert selection (in real system, this uses embeddings)
        selected_experts = simulate_expert_selection(
            test['question'], 
            test['user_profile'],
            test['sensitivity_score']
        )
        
        print(f"\nSelected {TOP_K_EXPERTS} Experts (from pool of {len(CULTURE_POOL)}):")
        for j, (culture, score) in enumerate(selected_experts[:TOP_K_EXPERTS], 1):
            print(f"  {j}. {culture} (similarity: {score:.3f})")
        
        # Show why these were selected
        print(f"\nSelection Rationale:")
        if test['user_profile']['location'] in [c for c, _ in selected_experts[:TOP_K_EXPERTS]]:
            print(f"  ✓ Included user's culture ({test['user_profile']['location']})")
        
        # Check for regional diversity
        regions = categorize_regions([c for c, _ in selected_experts[:TOP_K_EXPERTS]])
        print(f"  ✓ Regional diversity: {', '.join(set(regions.values()))}")
        
        # Show what would happen with different K values
        print(f"\nImpact of K value:")
        for k in [3, 5, 7]:
            selected_k = [c for c, _ in selected_experts[:k]]
            print(f"  K={k}: {', '.join(selected_k)}")

def simulate_expert_selection(question: str, user_profile: Dict, sensitivity_score: int) -> List[tuple]:
    """
    Simulate expert selection based on relevance.
    In the real system, this uses embedding similarity.
    """
    # Simulate similarity scores (in reality, these come from embedding comparisons)
    np.random.seed(hash(question) % 1000)  # Deterministic for demo
    
    scores = []
    for culture in CULTURE_POOL:
        # Base random score
        base_score = np.random.uniform(0.5, 0.9)
        
        # Boost score for user's culture
        if culture == user_profile.get('location'):
            base_score += 0.15
        
        # Boost score for culturally related regions
        if is_culturally_related(culture, user_profile):
            base_score += 0.1
        
        # Adjust based on sensitivity (more sensitive = more diverse selection)
        diversity_factor = sensitivity_score / 10 * 0.1
        base_score += np.random.uniform(-diversity_factor, diversity_factor)
        
        scores.append((culture, min(base_score, 1.0)))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def is_culturally_related(culture: str, user_profile: Dict) -> bool:
    """Check if a culture is related to user's background."""
    regions = {
        "East Asian": ["China", "Japan", "Vietnam", "Thailand"],
        "South Asian": ["India"],
        "Western": ["United States", "Germany", "France", "Italy", "Spain"],
        "Middle Eastern": ["Turkey", "Egypt"],
        "African": ["South Africa", "Kenya", "Nigeria"],
        "Latin American": ["Brazil", "Mexico"],
        "Southeast Asian": ["Indonesia", "Philippines", "Thailand", "Vietnam"],
        "Eastern European": ["Russia"]
    }
    
    user_bg = user_profile.get('cultural_background', '')
    for region, countries in regions.items():
        if region == user_bg and culture in countries:
            return True
    return False

def categorize_regions(cultures: List[str]) -> Dict[str, str]:
    """Categorize cultures by region."""
    region_map = {
        "United States": "North America",
        "China": "East Asia",
        "India": "South Asia",
        "Japan": "East Asia",
        "Turkey": "Middle East",
        "Vietnam": "Southeast Asia",
        "Russia": "Eastern Europe",
        "Brazil": "South America",
        "South Africa": "Africa",
        "Germany": "Western Europe",
        "France": "Western Europe",
        "Italy": "Southern Europe",
        "Spain": "Southern Europe",
        "Mexico": "Latin America",
        "Egypt": "Middle East",
        "Kenya": "East Africa",
        "Nigeria": "West Africa",
        "Indonesia": "Southeast Asia",
        "Philippines": "Southeast Asia",
        "Thailand": "Southeast Asia"
    }
    return {c: region_map.get(c, "Unknown") for c in cultures}

def demonstrate_configuration_impact():
    """Show how different configurations affect the system."""
    print("\n\n" + "=" * 80)
    print("CONFIGURATION IMPACT ANALYSIS")
    print("=" * 80)
    
    configs = [
        {"name": "Conservative", "k": 3, "pool_size": 6},
        {"name": "Balanced", "k": 5, "pool_size": 20},
        {"name": "Comprehensive", "k": 7, "pool_size": 20},
        {"name": "Maximum Diversity", "k": 10, "pool_size": 20}
    ]
    
    print(f"\n{'Config Name':<20} {'K':<5} {'Pool':<6} {'Coverage':<10} {'Avg Response Time*':<20}")
    print("-" * 70)
    
    for config in configs:
        coverage = (config['k'] / config['pool_size']) * 100
        # Estimate response time (more experts = longer time)
        base_time = 15  # Base sensitivity analysis time
        expert_time = config['k'] * 3  # 3 seconds per expert
        total_time = base_time + expert_time
        
        print(f"{config['name']:<20} {config['k']:<5} {config['pool_size']:<6} "
              f"{coverage:>6.1f}%    {total_time:>3}s")
    
    print("\n* Estimated with 3s per expert consultation")
    
    print("\n\nRECOMMENDATIONS:")
    print("- For general use: K=5 with 20 culture pool (balanced diversity)")
    print("- For speed priority: K=3 with smaller pool")
    print("- For maximum cultural coverage: K=7-10 with full pool")
    print("- Consider caching frequent culture combinations")

if __name__ == "__main__":
    # Run demonstrations
    demonstrate_expert_selection()
    demonstrate_configuration_impact()
    
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Current system uses only 6 cultures (should expand to 20)")
    print(f"✓ Selects top K={TOP_K_EXPERTS} experts based on embedding similarity")
    print(f"✓ Selection considers: topic relevance, user profile, cultural diversity")
    print(f"✓ Parallel processing speeds up expert consultations")
    print(f"✓ Caching prevents redundant expert queries")