import pytest
from src.nodes import (
    determine_cultural_sensitivity,
    extract_sensitive_topics,
    route_to_cultures,
    cultural_expert_node_factory,
    judge_agent
)
from src.types import GraphState

@pytest.fixture
def basic_state():
    return GraphState({
        "question_meta": {
            "original": "What are appropriate wedding customs?"
        },
        "user_profile": {
            "id": "test_user",
            "demographics": {"region": "Global"},
            "preferences": {}
        },
        "response_state": {
            "expert_responses": []
        },
        "full_history": []
    })

def test_cultural_sensitivity_detection(basic_state):
    result = determine_cultural_sensitivity(basic_state)
    assert "question_meta" in result
    assert "is_sensitive" in result["question_meta"]
    assert isinstance(result["question_meta"]["is_sensitive"], bool)

def test_topic_extraction(basic_state):
    result = extract_sensitive_topics(basic_state)
    assert "question_meta" in result
    assert "sensitive_topics" in result["question_meta"]
    assert isinstance(result["question_meta"]["sensitive_topics"], list)

def test_culture_routing(basic_state):
    result = route_to_cultures(basic_state)
    assert "question_meta" in result
    assert "relevant_cultures" in result["question_meta"]
    assert isinstance(result["question_meta"]["relevant_cultures"], list)

def test_cultural_expert_node(basic_state):
    expert_node = cultural_expert_node_factory("TestCulture")
    result = expert_node(basic_state)
    assert "response_state" in result
    assert "expert_responses" in result["response_state"]
    assert isinstance(result["response_state"]["expert_responses"], list)

def test_judge_agent(basic_state):
    # Add some expert responses first
    state_with_responses = {
        **basic_state,
        "response_state": {
            "expert_responses": [
                {"culture": "Culture1", "response": "Response1"},
                {"culture": "Culture2", "response": "Response2"}
            ]
        }
    }
    result = judge_agent(state_with_responses)
    assert "response_state" in result
    assert "judged" in result["response_state"]
    assert isinstance(result["response_state"]["judged"], str)