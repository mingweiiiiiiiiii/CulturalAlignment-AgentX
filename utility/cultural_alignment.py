"""
Cultural Alignment Utility Module

This module provides functionality for:
1. Deriving relevant cultures from user profiles
2. Calculating meaningful cultural alignment scores

The alignment score measures how well selected experts match the user's cultural context,
avoiding the circular logic where relevant_cultures = selected_cultures.
"""

from typing import Dict, List, Set, Any

# Mapping of locations/ethnicities to cultural groups
LOCATION_TO_CULTURES = {
    # US States
    "California/CA": ["United States"],
    "Texas/TX": ["United States"],
    "New York/NY": ["United States"],
    "Florida/FL": ["United States"],
    "Illinois/IL": ["United States"],
    "Pennsylvania/PA": ["United States"],
    "Ohio/OH": ["United States"],
    "Georgia/GA": ["United States"],
    "North Carolina/NC": ["United States"],
    "Michigan/MI": ["United States"],
    "New Jersey/NJ": ["United States"],
    "Virginia/VA": ["United States"],
    "Washington/WA": ["United States"],
    "Arizona/AZ": ["United States"],
    "Massachusetts/MA": ["United States"],
    "Tennessee/TN": ["United States"],
    "Indiana/IN": ["United States"],
    "Missouri/MO": ["United States"],
    "Maryland/MD": ["United States"],
    "Wisconsin/WI": ["United States"],
    "Colorado/CO": ["United States"],
    "Minnesota/MN": ["United States"],
    "South Carolina/SC": ["United States"],
    "Alabama/AL": ["United States"],
    "Louisiana/LA": ["United States"],
    "Kentucky/KY": ["United States"],
    "Oregon/OR": ["United States"],
    "Oklahoma/OK": ["United States"],
    "Connecticut/CT": ["United States"],
    "Utah/UT": ["United States"],
    "Iowa/IA": ["United States"],
    "Nevada/NV": ["United States"],
    "Arkansas/AR": ["United States"],
    "Mississippi/MS": ["United States"],
    "Kansas/KS": ["United States"],
    "New Mexico/NM": ["United States"],
    "Nebraska/NE": ["United States"],
    "West Virginia/WV": ["United States"],
    "Idaho/ID": ["United States"],
    "Hawaii/HI": ["United States"],
    "New Hampshire/NH": ["United States"],
    "Maine/ME": ["United States"],
    "Montana/MT": ["United States"],
    "Rhode Island/RI": ["United States"],
    "Delaware/DE": ["United States"],
    "South Dakota/SD": ["United States"],
    "North Dakota/ND": ["United States"],
    "Alaska/AK": ["United States"],
    "Vermont/VT": ["United States"],
    "Wyoming/WY": ["United States"],
    "District of Columbia/DC": ["United States"],
    
    # Countries
    "China": ["China"],
    "India": ["India"],
    "Indonesia": ["Indonesia"],
    "Pakistan": ["Pakistan"],
    "Brazil": ["Brazil"],
    "Nigeria": ["Nigeria"],
    "Bangladesh": ["Bangladesh"],
    "Russia": ["Russia"],
    "Mexico": ["United States", "Mexico"],  # Mexico borders US, cultural overlap
    "Japan": ["Japan"],
    "Ethiopia": ["Ethiopia"],
    "Philippines": ["Philippines"],
    "Egypt": ["Egypt"],
    "Vietnam": ["Vietnam"],
    "Iran": ["Iran"],
    "Turkey": ["Turkey"],
    "Germany": ["Germany"],
    "Thailand": ["Thailand", "Vietnam"],  # Regional neighbors
    "United Kingdom": ["United Kingdom"],
    "France": ["France"],
    "Italy": ["Italy"],
    "South Africa": ["South Africa"],
    "Myanmar": ["Myanmar", "India", "China"],  # Cultural influences
    "South Korea": ["South Korea", "Japan"],  # Cultural similarities
    "Colombia": ["Colombia", "Brazil"],  # Regional neighbors
    "Spain": ["Spain"],
    "Ukraine": ["Ukraine", "Russia"],  # Historical ties
    "Argentina": ["Argentina", "Brazil"],  # Regional neighbors
    "Kenya": ["Kenya", "Ethiopia"],  # Regional neighbors
    "Poland": ["Poland", "Germany", "Russia"],  # Historical/geographic ties
    "Canada": ["United States", "France"],  # Cultural influences
    "Uganda": ["Uganda", "Kenya"],  # Regional neighbors
    "Iraq": ["Iraq", "Turkey", "Iran"],  # Regional context
    "Afghanistan": ["Afghanistan", "Pakistan", "Iran"],  # Regional context
    "Morocco": ["Morocco", "France"],  # Historical ties
    "Saudi Arabia": ["Saudi Arabia", "Egypt"],  # Regional/cultural ties
    "Uzbekistan": ["Uzbekistan", "Russia"],  # Historical ties
    "Peru": ["Peru", "Brazil", "Colombia"],  # Regional neighbors
    "Malaysia": ["Malaysia", "Indonesia"],  # Cultural similarities
    "Venezuela": ["Venezuela", "Colombia", "Brazil"],  # Regional neighbors
    "Nepal": ["Nepal", "India", "China"],  # Geographic/cultural ties
    "Yemen": ["Yemen", "Saudi Arabia"],  # Regional ties
    "Ghana": ["Ghana", "Nigeria"],  # Regional neighbors
    "Mozambique": ["Mozambique", "South Africa"],  # Regional neighbors
    "Australia": ["United Kingdom", "United States"],  # Cultural heritage
    "North Korea": ["China", "Russia"],  # Political/cultural ties
    "Taiwan": ["China", "Japan"],  # Cultural influences
    "Syria": ["Syria", "Turkey", "Egypt"],  # Regional context
    "Ivory Coast": ["Ivory Coast", "Ghana"],  # Regional neighbors
    "Madagascar": ["Madagascar", "France"],  # Historical ties
    "Cameroon": ["Cameroon", "Nigeria"],  # Regional neighbors
    "Sri Lanka": ["Sri Lanka", "India"],  # Cultural ties
    "Burkina Faso": ["Burkina Faso", "Ghana"],  # Regional neighbors
}

# Ancestry/ethnicity to cultures mapping
ANCESTRY_TO_CULTURES = {
    "European": ["United States", "Germany", "France", "Italy", "Spain", "United Kingdom"],
    "Asian": ["China", "India", "Japan", "South Korea", "Vietnam", "Philippines"],
    "African": ["Nigeria", "Kenya", "Ghana", "South Africa", "Ethiopia"],
    "Hispanic": ["Mexico", "Colombia", "Argentina", "Peru", "Venezuela"],
    "Middle Eastern": ["Turkey", "Egypt", "Saudi Arabia", "Iran", "Iraq"],
    "Japanese": ["Japan"],
    "Chinese": ["China"],
    "Indian": ["India"],
    "Korean": ["South Korea"],
    "Mexican": ["Mexico"],
    "Filipino": ["Philippines"],
    "Vietnamese": ["Vietnam"],
    "German": ["Germany"],
    "Irish": ["United Kingdom"],
    "English": ["United Kingdom"],
    "Italian": ["Italy"],
    "Polish": ["Poland"],
    "French": ["France"],
    "Russian": ["Russia"],
    "Arab": ["Egypt", "Saudi Arabia", "Iraq", "Syria"],
    "Jewish": ["United States", "Russia"],  # Diaspora cultures
    "African American": ["United States", "Nigeria", "Ghana"],  # Heritage connections
    "Native American": ["United States", "Mexico"],  # Indigenous connections
    "Brazilian": ["Brazil"],
    "Turkish": ["Turkey"],
    "Iranian": ["Iran"],
    "Pakistani": ["Pakistan"],
    "Bangladeshi": ["Bangladesh"],
    "Ethiopian": ["Ethiopia"],
    "Nigerian": ["Nigeria"],
    "Egyptian": ["Egypt"],
    "Kenyan": ["Kenya"],
    "South African": ["South Africa"],
}

# Language to cultures mapping
LANGUAGE_TO_CULTURES = {
    "English": ["United States", "United Kingdom", "India", "Nigeria", "Philippines", "South Africa"],
    "Spanish": ["Spain", "Mexico", "Colombia", "Argentina", "Peru", "Venezuela"],
    "Chinese": ["China"],
    "Hindi": ["India"],
    "Arabic": ["Egypt", "Saudi Arabia", "Iraq", "Syria", "Morocco"],
    "Portuguese": ["Brazil", "Mozambique"],
    "Russian": ["Russia", "Ukraine", "Uzbekistan"],
    "Japanese": ["Japan"],
    "German": ["Germany"],
    "French": ["France", "Ivory Coast", "Morocco", "Madagascar"],
    "Korean": ["South Korea"],
    "Vietnamese": ["Vietnam"],
    "Turkish": ["Turkey"],
    "Italian": ["Italy"],
    "Polish": ["Poland"],
    "Ukrainian": ["Ukraine"],
    "Dutch": ["Germany"],  # Cultural proximity
    "Greek": ["Turkey"],  # Historical connections
    "Hebrew": ["Turkey"],  # Regional context
    "Swahili": ["Kenya", "Uganda"],
    "Yoruba": ["Nigeria"],
    "Bengali": ["Bangladesh", "India"],
    "Punjabi": ["India", "Pakistan"],
    "Urdu": ["Pakistan", "India"],
    "Persian": ["Iran", "Afghanistan"],
    "Tagalog": ["Philippines"],
    "Indonesian": ["Indonesia"],
    "Malay": ["Malaysia", "Indonesia"],
    "Thai": ["Thailand"],
    "Burmese": ["Myanmar"],
}


def derive_relevant_cultures(user_profile: Dict[str, Any]) -> List[str]:
    """
    Derive cultures relevant to a user based on their profile.
    This is independent of the question and represents the user's cultural context.
    
    Args:
        user_profile: Dictionary containing user information (place of birth, ancestry, etc.)
        
    Returns:
        List of culture names relevant to this user (max 5)
    """
    relevant_cultures: Set[str] = set()
    
    # 1. Place of birth - primary culture
    place_of_birth = user_profile.get("place of birth", user_profile.get("place_of_birth", ""))
    if place_of_birth and place_of_birth not in ["N/A", "", None]:
        if place_of_birth in LOCATION_TO_CULTURES:
            relevant_cultures.update(LOCATION_TO_CULTURES[place_of_birth])
        else:
            # Try to match partial (for states without /XX)
            for loc, cultures in LOCATION_TO_CULTURES.items():
                if place_of_birth in loc or loc in place_of_birth:
                    relevant_cultures.update(cultures)
                    break
    
    # 2. Ancestry/ethnicity - heritage culture  
    ancestry = user_profile.get("ancestry", user_profile.get("ethnicity", ""))
    if ancestry and ancestry not in ["N/A", "", None]:
        if ancestry in ANCESTRY_TO_CULTURES:
            relevant_cultures.update(ANCESTRY_TO_CULTURES[ancestry])
        # Try partial matching for ancestry too
        else:
            for anc, cultures in ANCESTRY_TO_CULTURES.items():
                if anc.lower() in ancestry.lower() or ancestry.lower() in anc.lower():
                    relevant_cultures.update(cultures)
                    break
    
    # 3. Race (secondary indicator)
    race = user_profile.get("race", "")
    if race and race not in ["N/A", "", None]:
        # Map race to broad cultural groups
        if "Asian" in race:
            relevant_cultures.update(["China", "India", "Japan"])
        elif "Black" in race or "African" in race:
            relevant_cultures.update(["United States", "Nigeria"])
        elif "Hispanic" in race or "Latino" in race:
            relevant_cultures.update(["Mexico", "United States"])
    
    # 4. Language - linguistic culture
    language = user_profile.get("household language", user_profile.get("language", ""))
    if language and language not in ["N/A", "", None]:
        if language in LANGUAGE_TO_CULTURES:
            # Only add top 2 cultures for the language to avoid over-expansion
            lang_cultures = LANGUAGE_TO_CULTURES[language][:2]
            relevant_cultures.update(lang_cultures)
    
    # 5. Current location (if different from birthplace)
    current_loc = user_profile.get("current_location", user_profile.get("location", ""))
    if current_loc and current_loc not in ["N/A", "", None] and current_loc != place_of_birth:
        if current_loc in LOCATION_TO_CULTURES:
            relevant_cultures.update(LOCATION_TO_CULTURES[current_loc])
    
    # If no cultures identified, try to infer from any available information
    if not relevant_cultures:
        # Check if we have ANY location information
        for field in ["place of birth", "place_of_birth", "location", "current_location"]:
            loc = user_profile.get(field, "")
            if loc and loc not in ["N/A", "", None]:
                # Even if not exact match, try to find something
                for location, cultures in LOCATION_TO_CULTURES.items():
                    if loc.lower() in location.lower() or location.lower() in loc.lower():
                        relevant_cultures.update(cultures)
                        break
                if relevant_cultures:
                    break
        
        # If still nothing, default to US
        if not relevant_cultures:
            relevant_cultures.add("United States")
    
    # Convert to list and limit to max 5 most relevant cultures
    relevant_list = list(relevant_cultures)[:5]
    
    return relevant_list


def calculate_meaningful_alignment(expert_responses: Dict[str, Any], 
                                 selected_cultures: List[str], 
                                 relevant_cultures: List[str]) -> float:
    """
    Calculate a meaningful cultural alignment score.
    
    Alignment measures how well the selected experts match the user's cultural context.
    - 1.0 = All consulted experts are from cultures relevant to the user
    - 0.0 = No consulted experts are from cultures relevant to the user
    
    Args:
        expert_responses: Dictionary of expert responses by culture
        selected_cultures: Cultures selected by the router (not used for alignment)
        relevant_cultures: Cultures relevant to the user (from their profile)
        
    Returns:
        Alignment score between 0.0 and 1.0
    """
    if not expert_responses or not relevant_cultures:
        return 0.0
    
    # Get cultures that provided full responses
    response_cultures = [
        culture for culture, info in expert_responses.items()
        if info.get('response_type') == 'full'
    ]
    
    if not response_cultures:
        return 0.0
    
    # Count how many response cultures are in the relevant set
    aligned = [c for c in response_cultures if c in relevant_cultures]
    
    # Alignment = proportion of responses from relevant cultures
    alignment_score = len(aligned) / len(response_cultures)
    
    return alignment_score


def test_cultural_alignment():
    """Test function to validate cultural alignment logic."""
    # Test cases
    test_profiles = [
        {
            "place of birth": "California/CA",
            "ancestry": "Mexican",
            "household language": "Spanish",
            "race": "Hispanic"
        },
        {
            "place of birth": "India",
            "ethnicity": "Indian",
            "household language": "Hindi"
        },
        {
            "place of birth": "New York/NY",
            "ancestry": "Jewish",
            "household language": "English",
            "race": "White alone"
        },
        {
            "place of birth": "Nigeria",
            "ethnicity": "Nigerian",
            "household language": "English"
        },
        {
            "place of birth": "Japan",
            "ancestry": "Japanese",
            "household language": "Japanese"
        }
    ]
    
    print("Testing cultural relevance derivation:")
    print("="*60)
    
    for i, profile in enumerate(test_profiles):
        relevant = derive_relevant_cultures(profile)
        print(f"\nProfile {i+1}:")
        print(f"  Birth: {profile.get('place of birth', 'N/A')}")
        print(f"  Ancestry: {profile.get('ancestry', profile.get('ethnicity', 'N/A'))}")
        print(f"  Language: {profile.get('household language', 'N/A')}")
        print(f"  → Relevant cultures: {relevant}")
    
    print("\n" + "="*60)
    print("Alignment scoring examples:")
    
    # Example 1: Good alignment
    expert_responses = {
        "United States": {"response_type": "full"},
        "Mexico": {"response_type": "full"},
        "India": {"response_type": "full"}
    }
    selected = ["United States", "Mexico", "India", "China", "Japan"]
    relevant = ["United States", "Mexico"]
    score = calculate_meaningful_alignment(expert_responses, selected, relevant)
    print(f"\nSelected experts from: {selected[:3]}")
    print(f"User's relevant cultures: {relevant}")
    print(f"Alignment score: {score:.2f} (2/3 experts match user's context)")
    
    # Example 2: Poor alignment
    expert_responses = {
        "China": {"response_type": "full"},
        "Japan": {"response_type": "full"},
        "South Korea": {"response_type": "full"}
    }
    relevant = ["United States", "Mexico"]
    score = calculate_meaningful_alignment(expert_responses, selected, relevant)
    print(f"\nSelected experts from: China, Japan, South Korea")
    print(f"User's relevant cultures: {relevant}")
    print(f"Alignment score: {score:.2f} (0/3 experts match user's context)")


if __name__ == "__main__":
    test_cultural_alignment()