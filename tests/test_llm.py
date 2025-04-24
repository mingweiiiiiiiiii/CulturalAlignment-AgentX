import pytest
from unittest.mock import Mock, patch
from llm_clients import (
    get_llm_client,
    LLMResponse,
    format_chat_message
)
from llm_manager import LLMManager
from ..llm_clients import GeminiClient, GroqClient, HuggingFaceClient, LamdaClient

def test_llm_client_initialization():
    client = get_llm_client()
    assert client is not None
    assert hasattr(client, 'generate')
    assert hasattr(client, 'get_embedding')

@pytest.fixture
def mock_llm_client():
    mock_client = Mock()
    mock_client.generate.return_value = "Test response"
    mock_client.get_embedding.return_value = [0.1] * 768
    return mock_client

def test_llm_response_formatting():
    response = LLMResponse(
        content="Test content",
        role="assistant",
        metadata={"temperature": 0.7}
    )
    
    assert response.content == "Test content"
    assert response.role == "assistant"
    assert response.metadata["temperature"] == 0.7

def test_chat_message_formatting():
    message = format_chat_message(
        role="user",
        content="Test message",
        name="TestUser"
    )
    
    assert message["role"] == "user"
    assert message["content"] == "Test message"
    assert message["name"] == "TestUser"

def test_llm_manager(mock_llm_client):
    manager = LLMManager(llm_client=mock_llm_client)
    
    # Test generation
    response = manager.generate("Test prompt")
    assert isinstance(response, str)
    assert response == "Test response"
    
    # Test embedding
    embedding = manager.get_embedding("Test text")
    assert len(embedding) == 768
    assert all(isinstance(x, float) for x in embedding)

@patch('llm_clients.get_llm_client')
def test_llm_error_handling(mock_get_client):
    mock_client = Mock()
    mock_client.generate.side_effect = Exception("API Error")
    mock_get_client.return_value = mock_client
    
    manager = LLMManager()
    
    # Test error handling during generation
    with pytest.raises(Exception):
        manager.generate("Test prompt")

def test_llm_retries(mock_llm_client):
    # Configure mock to fail twice then succeed
    responses = [
        Exception("First failure"),
        Exception("Second failure"),
        "Success"
    ]
    mock_llm_client.generate.side_effect = responses
    
    manager = LLMManager(llm_client=mock_llm_client)
    response = manager.generate("Test prompt", max_retries=3)
    
    assert response == "Success"
    assert mock_llm_client.generate.call_count == 3

def test_llm_temperature_settings():
    manager = LLMManager()
    
    # Test different temperature settings
    creative_response = manager.generate(
        "Test prompt",
        temperature=0.8
    )
    assert isinstance(creative_response, str)
    
    precise_response = manager.generate(
        "Test prompt",
        temperature=0.2
    )
    assert isinstance(precise_response, str)

def test_concurrent_llm_requests(mock_llm_client):
    manager = LLMManager(llm_client=mock_llm_client)
    
    # Test concurrent processing
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(5):
            futures.append(
                executor.submit(manager.generate, "Test prompt")
            )
        
        responses = [f.result() for f in futures]
    
    assert len(responses) == 5
    assert all(isinstance(r, str) for r in responses)

def test_prompt_validation():
    manager = LLMManager()
    
    # Test empty prompt
    with pytest.raises(ValueError):
        manager.generate("")
    
    # Test None prompt
    with pytest.raises(ValueError):
        manager.generate(None)
    
    # Test very long prompt
    long_prompt = "test " * 10000
    with pytest.raises(ValueError):
        manager.generate(long_prompt)

def test_response_validation(mock_llm_client):
    # Configure mock to return invalid responses
    invalid_responses = [
        "",  # Empty response
        None,  # None response
        "   ",  # Whitespace response
    ]
    
    for invalid_response in invalid_responses:
        mock_llm_client.generate.return_value = invalid_response
        manager = LLMManager(llm_client=mock_llm_client)
        
        with pytest.raises(ValueError):
            manager.generate("Test prompt")