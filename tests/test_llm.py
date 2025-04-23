import pytest
from ..llm_manager import LLMManager
from ..llm_clients import GeminiClient, GroqClient, HuggingFaceClient, LamdaClient

@pytest.fixture
def llm_manager():
    return LLMManager(model_name="gemini")

def test_llm_manager_initialization():
    # Test with valid model name
    manager = LLMManager(model_name="gemini")
    assert manager is not None
    
    # Test with invalid model name
    with pytest.raises(ValueError):
        LLMManager(model_name="invalid_model")
    
    # Test singleton behavior
    with pytest.raises(Exception):
        LLMManager(model_name="gemini")

def test_llm_manager_get_instance():
    # Test getting singleton instance
    manager = LLMManager.get_instance()
    assert manager is not None
    assert isinstance(manager, LLMManager)

def test_gemini_client():
    client = GeminiClient()
    response = client.generate("Test prompt")
    assert isinstance(response, str)
    assert len(response) > 0

def test_groq_client():
    client = GroqClient()
    response = client.generate("Test prompt")
    assert isinstance(response, str)
    assert len(response) > 0

def test_huggingface_client():
    client = HuggingFaceClient()
    response = client.generate("Test prompt")
    assert isinstance(response, str)
    assert len(response) > 0

def test_lamda_client():
    client = LamdaClient()
    response = client.generate("Test prompt")
    assert isinstance(response, str)
    assert len(response) > 0

def test_error_handling():
    # Test invalid API key
    with pytest.raises(Exception):
        client = GeminiClient(api_key="invalid_key")
        client.generate("Test prompt")
    
    # Test empty prompt
    client = GeminiClient()
    with pytest.raises(ValueError):
        client.generate("")
    
    # Test very long prompt
    long_prompt = "test " * 10000
    response = client.generate(long_prompt)
    assert isinstance(response, str)

def test_concurrent_requests():
    client = GeminiClient()
    import concurrent.futures
    
    prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(client.generate, prompt) for prompt in prompts]
        responses = [f.result() for f in futures]
    
    assert len(responses) == len(prompts)
    assert all(isinstance(r, str) for r in responses)