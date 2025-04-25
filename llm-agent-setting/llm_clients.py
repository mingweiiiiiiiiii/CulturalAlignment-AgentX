import os
import requests
from typing import Optional

import google.generativeai as genai
from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq

from api_key import germini_api_key, groq_api_key, hf_api_key, lamda_api_key


class GeminiClient:
    def __init__(self, model_name='gemini-pro', api_key: Optional[str] = None):
        self.api_key = api_key or germini_api_key
        os.environ['GOOGLE_API_KEY'] = self.api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
        response = self.model.generate_content(prompt)
        return response.text


class GroqClient:
    def __init__(self, model_name='llama-3-8b-8192', api_key: Optional[str] = None, temperature: float = 0.5):
        self.api_key = api_key or groq_api_key
        os.environ["GROQ_API_KEY"] = self.api_key
        self.model = ChatGroq(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature,
            verbose=True
        )

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
        response = self.model.invoke(prompt)
        return response.content.strip()


class HuggingFaceClient:
    def __init__(self, model_name="deepseek-ai/DeepSeek-V3-0324", provider="novita", api_key: Optional[str] = None):
        self.api_key = api_key or hf_api_key
        self.client = InferenceClient(provider=provider, api_key=self.api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"].strip()


class LamdaClient:
    def __init__(self, model: str = "llama2-13b-chat", api_key: Optional[str] = None):
        self.api_key = api_key or lamda_api_key
        self.model = model
        self.endpoint = "https://api.lambdalabs.com/v1/completions"

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature
        }

        response = requests.post(self.endpoint, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        return result.get("completion", "").strip()
