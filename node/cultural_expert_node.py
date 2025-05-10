from abc import ABC
from typing import Dict
import unittest
from llmagentsetting import llm_clients

# === Cultural Expert Base Class ===
class CulturalExpert(ABC):
    def __init__(self, culture_name: str, country_name: str, state: Dict = None):
        self.culture_name = culture_name
        self.country_name = country_name
        self.client = llm_clients.OllamaClient()  # Pass state if OllamaClient is adapted to use it

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
        return self.client.generate(prompt, options={"num_predict": 150})  # Adjust num_predict as needed

    def __call__(self, state: Dict) -> str:
        question = state["question_meta"]["original"]
        enhanced_prompt = self.enhance_prompt(question)  # Use enhanced prompt
        response_text = self.client.generate(enhanced_prompt, options={"num_predict": 150})  # Adjust num_predict
        return response_text


# === Manager Class for Experts ===
class CulturalExpertManager:
    def __init__(self, state: Dict = None):
        self.state = state
        self.expert_instances = {}

    def generate_expert_instances(self):
        countries = ["United States", "China", "India", "Japan", "Turkey", "Vietnam"]
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

    def test_expert_instance_count(self):
        self.assertEqual(self.manager.number_of_experts(), 20)

    def test_inference_response_type(self):
        response = self.manager.inference("India", "How do people celebrate festivals?")
        self.assertIsInstance(response, str)

    def test_get_expert_valid(self):
        expert = self.manager.get_expert("France")
        self.assertIsNotNone(expert)

    def test_get_expert_invalid(self):
        expert = self.manager.get_expert("Atlantis")  # Non-existent country
        self.assertIsNone(expert)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
