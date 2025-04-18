import os
from abc import ABC, abstractmethod


from dotenv import load_dotenv


# Abstract base class for all LLM providers
class LLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt"""
        pass

    def __call__(self, prompt: str) -> str:
        """Make the provider callable like a function"""
        return self.generate(prompt)

# Gemini implementation


class GeminiLLMProvider(LLMProvider):
    """Google Gemini LLM provider with LangChain integration"""

    def __init__(
        self,
        api_key = None,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        max_tokens = None,
        top_p = None,
        top_k = None,
        **kwargs
    ):
        # Load environment variables from .env file
        load_dotenv()

        # Get API key from parameter, env var, or .env file
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY in .env file or environment variable")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.extra_params = kwargs

        # Initialize LangChain client
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.langchain_llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                **self.extra_params
            )
        except ImportError:
            raise ImportError(
                "Please install required packages: pip install langchain-google-genai python-dotenv")

    def generate(self, prompt: str) -> str:
        """Generate a response directly from a prompt string"""
        response = self.langchain_llm.invoke([("human", prompt)])
        return response.content

    def invoke(self, messages):
        """Direct access to LangChain's invoke method for message-based input"""
        return self.langchain_llm.invoke(messages)

# Mock implementation


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""

    def generate(self, prompt: str) -> str:
        """Generate mock responses based on prompt content"""
        print(f"Mock Model prompt: {prompt}")
        if "sensitivity score" in prompt:
            return "7"
        elif "components" in prompt:
            return "religion, gender"
        elif "which 2-3 cultures" in prompt:
            return "US, China"
        elif "representative" in prompt:
            return f"This is a {prompt.split()[4]} perspective."
        else:
            return "Aggregated culturally respectful answer."

# Factory function to create providers


def create_llm_provider(provider_type: str = "auto", **kwargs) -> LLMProvider:
    """
    Create an LLM provider of the specified type

    Args:
        provider_type: "auto", "gemini", or "mock"
        **kwargs: Additional parameters for the provider

    Returns:
        An LLMProvider instance
    """
    # Auto-detect if we should use Gemini or mock
    if provider_type == "auto":
        load_dotenv()
        provider_type = "gemini" if os.environ.get(
            "GOOGLE_API_KEY") else "mock"

    # Create the appropriate provider
    if provider_type == "gemini":
        try:
            return GeminiLLMProvider(**kwargs)
        except Exception as e:
            print(
                f"Warning: Failed to initialize Gemini: {e}. Using mock provider instead.")
            return MockLLMProvider()
    elif provider_type == "mock":
        return MockLLMProvider()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")



if __name__ == "__main__":
    model = create_llm_provider("auto")
    print(model("What is the capital of France?"))
    print(model("What is the sensitivity score of the question: 'What is the capital of France?'?"))
