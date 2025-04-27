import os
import requests
from typing import Optional

from google import genai
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from api_key import germini_api_key, groq_api_key, hf_api_key, lamda_api_key


class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        load_dotenv('.env', override=True)
        self.api_key = api_key or os.getenv("GOOGLE_STUDIO_API") or germini_api_key
        if not self.api_key:
            raise ValueError("Google Studio API key must be provided or set in environment variables.")
        
        os.environ["GOOGLE_API_KEY"] = self.api_key

        self.model = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
      
        response = self.model.models.generate_content(
    model="gemini-2.0-flash",contents=prompt)
    

        return response.text


class GroqClient:
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: Optional[str] = None, temperature: float = 0.7):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or groq_api_key
        if not self.api_key:
            raise ValueError("Groq API key must be provided or set in environment variables.")
        
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


if __name__ == "__main__":
   # myModel = GeminiClient()
   # return_text = myModel.generate("Who are you")
   # print(return_text)

    myGroq = GroqClient()
    return_text = myGroq.generate("Who are you")
    print(return_text)
