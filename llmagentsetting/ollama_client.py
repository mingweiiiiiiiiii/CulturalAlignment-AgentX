import os
import ollama
import config
from typing import Optional

class OllamaClient:
    """Ollama client that mimics the interface of other LLM clients"""
    
    def __init__(self, model_name: str = "granite3.3:latest", host: Optional[str] = None):
        self.host = host or config.OLLAMA_HOST
        self.model_name = model_name
        self.client = ollama.Client(host=self.host)
    
    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """Get completion from Ollama model"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 8192,      # Optimized context window
                    "num_thread": 8,      # Use multiple threads
                    "num_gpu": 999,       # Use all available GPU layers
                }
            )
            
            # Extract response text
            if hasattr(response, 'response'):
                return response.response.strip()
            elif isinstance(response, dict) and 'response' in response:
                return response['response'].strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            print(f"Ollama error: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate(self, prompt: str) -> str:
        """Generate method for compatibility"""
        return self.get_completion(prompt)