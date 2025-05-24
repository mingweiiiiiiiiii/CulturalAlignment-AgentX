import pytest
import numpy as np
import subprocess  # type: ignore
import time  # type: ignore
import requests  # type: ignore
from nodes import (  # type: ignore
    get_text_embedding,
    USExpert,
    ChineseExpert,
    IndianExpert,
    determine_cultural_sensitivity,
    extract_sensitive_topics,
    route_to_cultures,
    compose_final_response,
    gen_prompt,
)


# Test text embedding
def test_get_text_embedding(mock_embedding, mocker):
    mocker.patch("nodes.get_text_embedding", return_value=mock_embedding)
    text = "This is a test sentence"
    embedding = get_text_embedding(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)


def test_text_embedding_error_cases():
    # Test empty string
    embedding = get_text_embedding("")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)

    # Test very long text
    long_text = "test " * 1000
    embedding = get_text_embedding(long_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)

    # Test special characters
    special_text = "!@#$%^&*()_+"
    embedding = get_text_embedding(special_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)


def test_prompt_generation():
    question = "What is your cultural perspective on this topic?"
    topic_mean = np.random.rand(768)
    prompt_library = [
        "Consider the cultural implications...",
        "From your cultural background...",
        "Taking into account cultural sensitivities...",
    ]

    prompt = gen_prompt(question, topic_mean, prompt_library)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert question in prompt
    assert prompt_library[0] in prompt


# Test cultural experts
@pytest.mark.parametrize(
    "expert_class,culture",
    [(USExpert, "US"), (ChineseExpert, "China"), (IndianExpert, "India")],
)
def test_cultural_experts(expert_class, culture, mocker):
    mocker.patch("nodes.model", return_value=f"Response from {culture}")
    expert = expert_class()
    assert expert.culture_name == culture

    response = expert.generate_response("What is the meaning of life?")
    assert isinstance(response, str)
    assert len(response) > 0
    assert culture in response


def test_cultural_expert_mock_responses(mocker):
    # Setup mock for LLM responses
    mock_responses = {
        "US": "From a US perspective...",
        "China": "From a Chinese perspective...",
        "India": "From an Indian perspective...",
    }
    mocker.patch("nodes.model", side_effect=lambda x: mock_responses[x.split()[4]])

    # Test each expert
    for expert_class, culture in [
        (USExpert, "US"),
        (ChineseExpert, "China"),
        (IndianExpert, "India"),
    ]:
        expert = expert_class()
        response = expert.generate_response("Test question")
        assert response == mock_responses[culture]


# Test sensitivity detection
def test_determine_cultural_sensitivity(mock_graph_state):
    result = determine_cultural_sensitivity(mock_graph_state)

    assert "question_meta" in result
    assert "is_sensitive" in result["question_meta"]
    assert "sensitivity_score" in result["question_meta"]
    assert isinstance(result["question_meta"]["sensitivity_score"], (int, float))
    assert 0 <= result["question_meta"]["sensitivity_score"] <= 10


def test_sensitivity_detection_edge_cases():
    # Test highly sensitive content
    state = {
        "question_meta": {
            "original": "Why do certain religious groups believe in extremist ideologies?"
        }
    }
    result = determine_cultural_sensitivity(state)
    assert result["question_meta"]["is_sensitive"] is True
    assert result["question_meta"]["sensitivity_score"] >= 8

    # Test neutral content
    state = {"question_meta": {"original": "What time do people usually have dinner?"}}
    result = determine_cultural_sensitivity(state)
    assert result["question_meta"]["is_sensitive"] is False
    assert result["question_meta"]["sensitivity_score"] <= 3


# Test topic extraction
def test_extract_sensitive_topics():
    state = {
        "question_meta": {
            "original": "Why do certain cultures have different food preferences?"
        }
    }
    result = extract_sensitive_topics(state)

    assert "question_meta" in result
    assert "sensitive_topics" in result["question_meta"]
    assert isinstance(result["question_meta"]["sensitive_topics"], list)
    assert len(result["question_meta"]["sensitive_topics"]) > 0


def test_topic_extraction_complex_cases():
    # Test multiple topics
    state = {
        "question_meta": {
            "original": "Why do different religions have different dietary restrictions and marriage customs?"
        }
    }
    result = extract_sensitive_topics(state)
    topics = result["question_meta"]["sensitive_topics"]
    assert len(topics) >= 2
    assert "religion" in [t.lower() for t in topics]

    # Test intersectional topics
    state = {
        "question_meta": {
            "original": "How do gender roles differ across various social classes and cultures?"
        }
    }
    result = extract_sensitive_topics(state)
    topics = result["question_meta"]["sensitive_topics"]
    assert len(topics) >= 2
    assert any("gender" in t.lower() for t in topics)
    assert any("class" in t.lower() for t in topics)


# Test cultural routing
def test_route_to_cultures(mock_graph_state, mock_embedding):
    culture_embeddings = np.random.rand(3, 768)
    result = route_to_cultures(
        mock_graph_state, ["US", "China", "India"], culture_embeddings
    )

    assert "question_meta" in result
    assert "relevant_cultures" in result["question_meta"]
    assert isinstance(result["question_meta"]["relevant_cultures"], list)
    assert all(
        c in ["US", "China", "India"]
        for c in result["question_meta"]["relevant_cultures"]
    )


# Test response composition
def test_compose_final_response():
    activate_set = [
        ("US", 0.5, "From US perspective..."),
        ("China", 0.3, "From Chinese perspective..."),
        ("India", 0.2, "From Indian perspective..."),
    ]

    state = {
        "user_profile": {"preferences": {}, "demographics": {"country": "US"}},
        "question_meta": {
            "original": "How do people celebrate festivals?",
            "sensitive_topics": ["cultural_practices"],
            "relevant_cultures": ["US", "China", "India"],
        },
    }

    result = compose_final_response(state, activate_set)

    assert "response_state" in result
    assert "final" in result["response_state"]
    assert isinstance(result["response_state"]["final"], str)
    assert len(result["response_state"]["final"]) > 0


def test_compose_final_response_variations():
    # Test with different weights
    activate_set = [
        ("US", 0.8, "Strong US perspective..."),
        ("China", 0.1, "Minor Chinese perspective..."),
        ("India", 0.1, "Minor Indian perspective..."),
    ]

    state = {
        "user_profile": {
            "preferences": {"style": "formal"},
            "demographics": {"country": "US"},
        },
        "question_meta": {
            "original": "How are important life events celebrated?",
            "sensitive_topics": ["cultural_practices"],
            "relevant_cultures": ["US", "China", "India"],
        },
    }

    result = compose_final_response(state, activate_set)
    assert "response_state" in result
    assert "final" in result["response_state"]

    # Test with equal weights
    activate_set_equal = [
        ("US", 0.33, "Equal US perspective..."),
        ("China", 0.33, "Equal Chinese perspective..."),
        ("India", 0.34, "Equal Indian perspective..."),
    ]

    result_equal = compose_final_response(state, activate_set_equal)
    assert "response_state" in result_equal
    assert "final" in result_equal["response_state"]
    assert result_equal["response_state"]["final"] != result["response_state"]["final"]


# Add imports at top for integration testing
import subprocess
import time
import requests


@pytest.fixture(scope="session", autouse=True)
def docker_compose_setup():
    # Spin up Docker Compose stack for Ollama and application
    subprocess.run(["docker-compose", "up", "-d"], check=True)
    # Wait for Ollama health
    for _ in range(30):
        try:
            r = requests.get("http://localhost:11434/api/version")
            if r.status_code == 200 and 'version' in r.text:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        pytest.skip("Ollama service not available")
    yield
    # Tear down containers
    subprocess.run(["docker-compose", "down"], check=True)


def test_embed_endpoint(docker_compose_setup):
    # Test /api/embed endpoint
    payload = {"model": "mxbai-embed-large", "input": ["test sentence"]}
    response = requests.post("http://localhost:8000/api/embed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    embeddings = data["embeddings"]
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768
    assert all(isinstance(x, float) for x in embeddings[0])


def test_generate_endpoint(docker_compose_setup):
    # Test /api/generate endpoint
    payload = {"model": "phi4", "prompt": "Hello"}
    response = requests.post("http://localhost:8000/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0


def test_ollama_ready_log(docker_compose_setup):
    # Optional: check for startup log in Ollama container logs
    logs = subprocess.check_output(["docker", "logs", "ollama-gpu"], shell=False).decode()
    assert "Ollama ready" in logs
