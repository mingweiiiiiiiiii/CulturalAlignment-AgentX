from abc import ABC
from typing import Dict, List, Tuple, Any
from google import genai
import unittest
from llmagentsetting import llm_clients
import numpy as np
from node.embed_utils import embed_persona

# === LLM Model Wrapper ===
# class LLMModel:
#     def __init__(self):
#         self.api_key = "AIzaSyAlMLq2h1YHKJgOm6hds2aHz_iWrByXacM"  # Your API Key
#         self.model_name = "gemini-2.0-flash"
#         self.client = genai.Client(api_key=self.api_key)

#     def generate(self, prompt: str) -> str:
#         response = self.client.models.generate_content(
#             model=self.model_name, contents=prompt
#         )
#         return response.text.strip()


# === Cultural Expert Base Class ===
class CulturalExpert(ABC):
    def __init__(self, culture_name: str, country_name: str, state: Dict = None):
        self.culture_name = culture_name
        self.country_name = country_name
        self.client = llm_clients.LambdaAPIClient(state=state)

    def enhance_prompt(self, question: str) -> str:
        return (
            f"You are a cultural expert from {self.country_name}, deeply familiar with its historical, social, moral, and traditional nuances. "
            f"Frame your answer considering the values, etiquette, common beliefs, communication styles, and societal norms typical of {self.country_name}. "
            f"Include aspects like community vs individualism, indirect vs direct communication, formality levels, views on authority, spirituality, family roles, and social relationships. "
            f"Be thoughtful, factual, respectful of diversity, and avoid generalizations or stereotypes. "
            f"Keep the response under 150 words.\n\n"
            f"Question: '{question}'"
        )

    def generate_response(self, question: str) -> str:
        prompt = self.enhance_prompt(question)
        return self.client.get_completion(prompt)

    def __call__(self, state: Dict) -> str:
        question = state["question_meta"]["original"]
        response_text = self.client.get_completion(question)
        return response_text


# === Manager Class for Experts ===
class CulturalExpertManager:
    def __init__(self, state: Dict = None):
        self.state = state
        self.expert_instances: Dict[str, CulturalExpert] = {}
        self._persona_embeddings: Tuple[List[str], np.ndarray] = None  # Cache for persona embeddings

    def generate_expert_instances(self):
        # Full 20 countries representing diverse cultural perspectives
        countries = [
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
        
        for country in countries:
            self.expert_instances[country] = CulturalExpert(
                culture_name=f"{country} Culture",
                country_name=country,
                state=self.state
            )
        return self.expert_instances

    def get_expert(self, country_name: str) -> CulturalExpert:
        return self.expert_instances.get(country_name)

    def number_of_experts(self) -> int:
        return len(self.expert_instances)

    def inference(self, country_name: str, question: str) -> str:
        expert = self.get_expert(country_name)
        if expert is None:
            raise ValueError(f"No expert found for country: {country_name}")
        if self.state is not None:
            self.state["question_meta"] = {"original": question}
        else:
            self.state = {"question_meta": {"original": question}}
        return expert(self.state)
    
    def get_persona_prompt(self, country_name: str) -> str:
        """
        Return the static soft‑prompt that describes this expert's cultural persona.
        This prompt will be embedded once at startup to produce E_j.
        """
        demographic = self.state.get("user_profile", {}) # since seperate demographic info is not provided
        # You can enrich this template however you like.
        return (
            f"You are a knowledgeable advisor from {country_name}. "
            f"Take into account the user's demographics: {demographic}. "
            f"Provide culturally informed guidance reflecting {country_name} values and norms."
        )
    def get_all_persona_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Returns a tuple (expert_list, embeddings_matrix),
        where embeddings_matrix[i] corresponds to expert_list[i].
        Caches on first call.
        """
        # Only build once
        if self._persona_embeddings is None:
            # Ensure we have instances so get_persona_prompt works
            names = list(self.generate_expert_instances().keys())

            # Build persona prompt texts
            texts = {n: self.get_persona_prompt(n) for n in names}

            # Embed each persona prompt
            embs = {
                n: embed_persona({"persona": texts[n]})
                for n in names
            }

            # Stack into (N, d) array
            matrix = np.stack([embs[n] for n in names])

            # Cache both ordering and matrix
            self._persona_embeddings = (names, matrix)

        return self._persona_embeddings

# === Example Usage ===
if __name__ == "__main__":
    # Initialize model

    # Initialize manager
    manager = CulturalExpertManager()

    # Generate experts
    expert_instances = manager.generate_expert_instances()

    # Print number of experts
    print(f"Total number of expert instances: {manager.number_of_experts()}")

    # Example inference with Kenya expert
    kenya_response = manager.inference(
        "Kenya", "What are important values when raising children?"
    )
    print(f"\nKenya Expert Response:\n{kenya_response}")


# === Unit Test Cases ===
class TestCulturalExpertManager(unittest.TestCase):
    def setUp(self):
        self.manager = CulturalExpertManager()
        self.manager.generate_expert_instances()

    def test_all_20_experts_created(self):
        self.assertEqual(self.manager.number_of_experts(), 20)

    def test_get_expert(self):
        us_expert = self.manager.get_expert("United States")
        self.assertIsNotNone(us_expert)
        self.assertEqual(us_expert.country_name, "United States")

    def test_persona_embeddings(self):
        expert_list, embeddings = self.manager.get_all_persona_embeddings()
        self.assertEqual(len(expert_list), 20)
        self.assertEqual(embeddings.shape[0], 20)


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