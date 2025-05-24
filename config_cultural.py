"""
Configuration for cultural alignment system.
"""

# Cultural Expert Selection Configuration
CULTURAL_EXPERT_CONFIG = {
    # Number of experts to select for consultation
    "TOP_K_EXPERTS": 5,  # Select top 5 most relevant experts from the pool
    
    # Embedding similarity weights
    "LAMBDA_TOPIC": 0.6,  # Weight for topic embedding similarity
    "LAMBDA_USER": 0.4,   # Weight for user profile embedding similarity
    
    # Fallback threshold
    "TAU_THRESHOLD": -30.0,  # If max score < tau, use clustering fallback
    
    # Available cultures (20 diverse cultures)
    "AVAILABLE_CULTURES": [
        "United States",  # North America
        "China",          # East Asia
        "India",          # South Asia
        "Japan",          # East Asia
        "Turkey",         # Middle East/Europe
        "Vietnam",        # Southeast Asia
        "Russia",         # Eastern Europe/Asia
        "Brazil",         # South America
        "South Africa",   # Africa
        "Germany",        # Western Europe
        "France",         # Western Europe
        "Italy",          # Southern Europe
        "Spain",          # Southern Europe
        "Mexico",         # North/Central America
        "Egypt",          # Middle East/Africa
        "Kenya",          # East Africa
        "Nigeria",        # West Africa
        "Indonesia",      # Southeast Asia
        "Philippines",    # Southeast Asia
        "Thailand"        # Southeast Asia
    ],
    
    # Clustering configuration for fallback
    "N_CLUSTERS": 5,  # Number of clusters for KMeans fallback
    
    # Response generation
    "MAX_EXPERT_RESPONSE_LENGTH": 150,  # Max words per expert response
    "PARALLEL_EXPERT_QUERIES": True,    # Query experts in parallel
    
    # Caching
    "ENABLE_CACHING": True,
    "CACHE_TTL_SECONDS": 3600,  # 1 hour cache TTL
}

# Sensitivity Detection Configuration
SENSITIVITY_CONFIG = {
    # Threshold for marking as culturally sensitive
    "SENSITIVITY_THRESHOLD": 5,  # Score >= 5 is considered sensitive
    
    # Topics that automatically trigger sensitivity
    "SENSITIVE_TOPICS": [
        "religion", "politics", "gender", "race", "ethnicity",
        "sexuality", "immigration", "nationalism", "cultural traditions",
        "family values", "social hierarchy", "economic systems"
    ]
}