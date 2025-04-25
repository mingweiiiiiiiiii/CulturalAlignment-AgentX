from llm_clients import GeminiClient, GroqClient, HuggingFaceClient, LamdaClient


class LLMManager:
    _instance = None  # Singleton instance (optional for extra safety)

    def __init__(self, model_name: str):
        if LLMManager._instance is not None:
            raise Exception("LLMManager is already initialized.")
        
        self.clients = {
            "gemini": GeminiClient(),
            "groq": GroqClient(),
            "huggingface": HuggingFaceClient(),
            "lamda": LamdaClient()
        }
        model_name = model_name.lower()
        if model_name not in self.clients:
            raise ValueError(f"Unsupported provider: {model_name}")
        self.model_name = model_name

        LLMManager._instance = self  # Store instance

    def generate(self, prompt: str) -> str:
        return self.clients[self.model_name].generate(prompt)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise Exception("LLMManager is not initialized yet.")
        return cls._instance

if __name__ == "__main__":
    # Initialize the LLMManager ONCE with your chosen provider
    llm = LLMManager(model_name="gemini")

    # Use it to generate text
    output = llm.generate("What's the future of AI?")
    print(output)