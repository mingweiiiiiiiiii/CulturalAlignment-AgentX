import os
from typing import Optional, Any, List, Dict
from mistralai import Mistral
from google import genai
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import requests

from .api_key import lambda_api_key

class GeminiClient:
    def __init__(
        self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None
    ):
        load_dotenv(".env", override=True)
        self.api_key = api_key or os.getenv("GOOGLE_STUDIO_API") 
        if not self.api_key:
            raise ValueError(
                "Google Studio API key must be provided or set in environment variables."
            )

        os.environ["GOOGLE_API_KEY"] = self.api_key

        self.model = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")

        response = self.model.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )

        return response.text


class GroqClient:
    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY") 
        if not self.api_key:
            raise ValueError(
                "Groq API key must be provided or set in environment variables."
            )

        os.environ["GROQ_API_KEY"] = self.api_key

        self.model = ChatGroq(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature,
            verbose=True,
        )

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")

        response = self.model.invoke(prompt)
        return response.content.strip()


class MistralClient:
    def __init__(
        self, model_name: str = "mistral-large-latest", api_key: Optional[str] = None
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key must be provided or set in environment variables."
            )

        os.environ["MISTRAL_API_KEY"] = self.api_key

        self.client = Mistral(api_key=self.api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")

        chat_response = self.client.chat.complete(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return chat_response.choices[0].message.content.strip()


class LambdaAPIClient:
    MODEL_LIST = [
        "deepseek-llama3.3-70b",
        "deepseek-r1-671b",
        "deepseek-v3-0324",
        "hermes3-405b",
        "hermes-3-llama-3.1-405b-fp8",
        "hermes3-70b",
        "hermes3-8b",
        "lfm-40b",
        "llama-4-maverick-17b-128e-instruct-fp8",
        "llama-4-scout-17b-16e-instruct",
        "llama3.1-405b-instruct-fp8",
        "llama3.1-70b-instruct-fp8",
        "llama3.1-8b-instruct",
        "llama3.1-nemotron-70b-instruct-fp8",
        "llama3.2-11b-vision-instruct",
        "llama3.2-3b-instruct",
        "llama3.3-70b-instruct-fp8",
        "qwen25-coder-32b-instruct"
    ]
    DEFAULT_MODEL = "qwen25-coder-32b-instruct"

    def __init__(self, api_key=None, germini_api_key=None):
        load_dotenv(".env", override=True)
        self.api_key = api_key or os.getenv("LAMBDA_API") or lambda_api_key

        if not self.api_key:
            raise ValueError(
                "Lambda API key must be provided or set in environment variables."
            )

        os.environ["LAMBDA_API"] = self.api_key

        self.url = "https://api.lambdalabs.com/v1/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_completion(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] = None,
    ) -> str:
        model = model or self.DEFAULT_MODEL
        if model not in self.MODEL_LIST:
            raise ValueError(f"Invalid model name. Choose from:\n{', '.join(self.MODEL_LIST)}")
        
        data: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop is not None:
            data["stop"] = stop
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
        
        response = requests.post(self.url, headers=self.headers, json=data)
        #print("Lambda API response:", response.status_code, response.text)
        response_data = response.json()

        try:
            return response_data["choices"][0]["text"].strip()
        except (KeyError, IndexError):
            return "Error: Unable to extract text from API response."

def test_lambdamain():
    prompt = input("Enter your prompt: ").strip()

    try:
        client = LambdaAPIClient()
        result = client.get_completion(prompt=prompt)
        print("Generated Text:", result)
    except ValueError as e:
        print("Error:", e)

if __name__ == "__main__":
    # myModel = GeminiClient()
    # return_text = myModel.generate("Who are you")
    # print(return_text)

    myGroq = GroqClient()
    return_text = myGroq.generate("Who are you")
    print(return_text)

    test_lambdamain()
