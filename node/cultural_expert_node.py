import numpy as np
import re
from abc import ABC, abstractmethod
from typing import Dict
from google import genai

# === LLM Model Wrapper ===
class LLMModel:
    def __init__(self):
        self.api_key = "AIzaSyAlMLq2h1YHKJgOm6hds2aHz_iWrByXacM"
        self.model_name = "gemini-2.0-flash"
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()

# === Cultural Expert Abstract Base Class ===
class CulturalExpert(ABC):
    def __init__(self, culture_name: str, model: LLMModel):
        self.culture_name = culture_name
        self.model = model

    @abstractmethod
    def enhance_prompt(self, question: str) -> str:
        pass

    def generate_response(self, question: str) -> str:
        prompt = self.enhance_prompt(question)
        return self.model.generate(prompt)

    def __call__(self, state) -> str:
        question = state["question_meta"]["original"]
        response_text = self.generate_response(question)
        return response_text

# === Cultural Expert Implementations ===
class USExpert(CulturalExpert):
    def __init__(self, model: LLMModel):
        super().__init__("US", model)

    def enhance_prompt(self, question: str) -> str:
        return (
            f"You are a cultural expert representing the United States. "
            f"Answer the following question in a way that reflects American values and perspectives. "
            f"Be concise, thoughtful, culturally specific, and respectful. "
            f"Your response must be within 100 words. Avoid hallucinations, fabrications, or offensive content.\n\n"
            f"Question: '{question}'"
        )

class ChineseExpert(CulturalExpert):
    def __init__(self, model: LLMModel):
        super().__init__("China", model)

    def enhance_prompt(self, question: str) -> str:
        return (
            f"You are a cultural expert representing China. "
            f"Answer the following question in a way that reflects Chinese values and perspectives. "
            f"Be concise, thoughtful, culturally specific, and respectful. "
            f"Your response must be within 100 words. Avoid hallucinations, fabrications, or offensive content.\n\n"
            f"Question: '{question}'"
        )

class IndianExpert(CulturalExpert):
    def __init__(self, model: LLMModel):
        super().__init__("India", model)

    def enhance_prompt(self, question: str) -> str:
        return (
            f"You are a cultural expert representing India. "
            f"Answer the following question in a way that reflects Indian values and perspectives. "
            f"Be concise, thoughtful, culturally specific, and respectful. "
            f"Your response must be within 100 words. Avoid hallucinations, fabrications, or offensive content.\n\n"
            f"Question: '{question}'"
        )

# === Example Usage Setup ===
if __name__ == "__main__":
    # Initialize model inside the class
    llm_model = LLMModel()

    # Create expert instances
    expert_classes = {
        "US": USExpert(model=llm_model),
        "China": ChineseExpert(model=llm_model),
        "India": IndianExpert(model=llm_model)
    }

    # Example input state
    input_state = {"question_meta": {"original": "How should conflicts be resolved in a community?"}}

    # Example expert call
    us_response_text = expert_classes["US"](input_state)
    print(us_response_text)
