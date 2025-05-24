from abc import ABC
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from llmagentsetting.ollama_client import OllamaClient
from node.embed_utils import embed_persona

class CulturalExpert(ABC):
    """Cultural expert that can assess relevance before generating full response."""
    
    def __init__(self, culture_name: str, country_name: str, state: Dict = None):
        self.culture_name = culture_name
        self.country_name = country_name
        self.state = state
        self.client = OllamaClient()
    
    def assess_cultural_relevance(self, question: str) -> Tuple[bool, float, str]:
        """
        Quickly assess if this question is culturally sensitive for this specific culture.
        Returns: (is_relevant, relevance_score, brief_reason)
        """
        prompt = f"""As a cultural expert from {self.country_name}, quickly assess if this question has specific cultural sensitivity or relevance to your culture.

Question: {question}

Rate from 0-10 where:
- 0-3: Not culturally specific (universal human experience)
- 4-6: Some cultural variation exists
- 7-10: Highly culturally specific/sensitive

Respond with JSON:
{{
    "score": <0-10>,
    "is_relevant": <true if score >= 5>,
    "reason": "<one sentence explanation>"
}}"""
        
        try:
            response = self.client.generate(prompt)
            import re
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return (
                    result.get("is_relevant", False),
                    result.get("score", 0),
                    result.get("reason", "No reason provided")
                )
        except Exception as e:
            print(f"Error assessing relevance for {self.country_name}: {e}")
        
        # Conservative fallback - assume relevant if assessment fails
        return True, 5.0, "Assessment failed, assuming relevance"
    
    def generate_full_response(self, question: str) -> str:
        """Generate full cultural perspective response."""
        prompt = f"""You are a cultural expert from {self.country_name}, deeply familiar with its values and perspectives.
        
Answer this question from your cultural viewpoint, considering:
- Traditional values and modern changes
- Common perspectives in your society
- How this differs from other cultures

Question: {question}

Provide a thoughtful response (100-150 words) that authentically represents {self.country_name} perspectives."""
        
        try:
            return self.client.generate(prompt)
        except Exception as e:
            return f"Error generating response from {self.country_name}: {str(e)}"
    
    def generate_brief_input(self, question: str) -> str:
        """Generate a brief cultural input when not highly relevant."""
        prompt = f"""As a {self.country_name} cultural expert, provide a very brief (1-2 sentences) perspective on:

{question}

Only mention if there's something notably different about {self.country_name}'s view."""
        
        try:
            return self.client.generate(prompt)
        except Exception as e:
            return f"No specific {self.country_name} perspective available."


class CulturalExpertManager:
    """Manages pool of 20 cultural experts with smart response generation."""
    
    def __init__(self, state: Dict = None):
        self.state = state
        self.expert_instances: Dict[str, CulturalExpert] = {}
        self._persona_embeddings: Tuple[List[str], np.ndarray] = None
        
        # Full 20 cultures representing global diversity
        self.cultures = [
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
        ]
    
    def generate_expert_instances(self):
        """Generate all 20 cultural expert instances."""
        for country in self.cultures:
            self.expert_instances[country] = CulturalExpert(
                culture_name=f"{country} Culture",
                country_name=country,
                state=self.state
            )
        return self.expert_instances
    
    def get_expert(self, country_name: str) -> Optional[CulturalExpert]:
        """Get a specific cultural expert."""
        if not self.expert_instances:
            self.generate_expert_instances()
        return self.expert_instances.get(country_name)
    
    def number_of_experts(self) -> int:
        """Get total number of experts in pool."""
        return len(self.cultures)
    
    def get_smart_expert_responses(self, question: str, selected_cultures: List[str], 
                                  relevance_threshold: float = 5.0) -> Dict[str, Dict[str, Any]]:
        """
        Get responses from selected experts, but only full responses if culturally relevant.
        
        Args:
            question: The question to answer
            selected_cultures: List of cultures selected by router (top K)
            relevance_threshold: Minimum score to generate full response
            
        Returns:
            Dict mapping culture to response info
        """
        responses = {}
        
        for culture in selected_cultures:
            expert = self.get_expert(culture)
            if not expert:
                continue
            
            # First, assess cultural relevance
            is_relevant, score, reason = expert.assess_cultural_relevance(question)
            
            # Generate appropriate response based on relevance
            if score >= relevance_threshold:
                # High relevance - generate full response
                response = expert.generate_full_response(question)
                response_type = "full"
            else:
                # Low relevance - generate brief input or skip
                response = expert.generate_brief_input(question)
                response_type = "brief"
            
            responses[culture] = {
                "response": response,
                "relevance_score": score,
                "is_relevant": is_relevant,
                "reason": reason,
                "response_type": response_type
            }
        
        return responses
    
    def get_persona_prompt(self, country_name: str) -> str:
        """Get embedding prompt for a cultural expert."""
        demographic = self.state.get("user_profile", {}) if self.state else {}
        
        # Rich cultural embedding prompt
        return (
            f"Cultural expert from {country_name} with deep knowledge of: "
            f"social values, family structures, religious beliefs, political systems, "
            f"economic philosophies, gender roles, generational relationships, "
            f"work culture, education values, and social hierarchies. "
            f"Represents modern {country_name} perspectives while understanding traditional values. "
            f"User context: {demographic}"
        )
    
    def get_all_persona_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Get cached embeddings for all cultural experts."""
        if self._persona_embeddings is None:
            if not self.expert_instances:
                self.generate_expert_instances()
            
            names = list(self.expert_instances.keys())
            texts = {n: self.get_persona_prompt(n) for n in names}
            
            # Embed each persona
            embs = {}
            for n in names:
                try:
                    embs[n] = embed_persona({"persona": texts[n]})
                except Exception as e:
                    print(f"Error embedding {n}: {e}")
                    embs[n] = np.zeros(768)  # Fallback embedding
            
            matrix = np.stack([embs[n] for n in names])
            self._persona_embeddings = (names, matrix)
        
        return self._persona_embeddings


# Example usage
if __name__ == "__main__":
    manager = CulturalExpertManager()
    manager.generate_expert_instances()
    
    print(f"Total experts in pool: {manager.number_of_experts()}")
    
    # Test smart response generation
    test_question = "What are your thoughts on arranged marriages?"
    selected = ["India", "United States", "Japan", "Brazil", "Germany"]
    
    print(f"\nTesting smart responses for: {test_question}")
    print(f"Selected cultures: {selected}")
    
    responses = manager.get_smart_expert_responses(test_question, selected)
    
    print("\nResults:")
    for culture, info in responses.items():
        print(f"\n{culture}:")
        print(f"  Relevance: {info['relevance_score']}/10")
        print(f"  Type: {info['response_type']}")
        print(f"  Reason: {info['reason']}")
        if info['response_type'] == 'brief':
            print(f"  Response: {info['response']}")