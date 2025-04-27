import pytest
from graph import create_cultural_graph
from custom_types import GraphState

def test_graph_creation():
    # Test with default cultures
    graph = create_cultural_graph()
    assert graph is not None

    # Test with custom cultures
    custom_cultures = ["US", "Japan", "India"]
    graph = create_cultural_graph(custom_cultures)
    assert graph is not None

def test_graph_execution(mock_graph_state):
    graph = create_cultural_graph()
    result = graph.run(GraphState(**mock_graph_state))
    
    assert result is not None
    assert "response_state" in result
    assert "final" in result["response_state"]

def test_graph_error_handling():
    graph = create_cultural_graph()
    
    # Test with invalid state
    with pytest.raises(Exception):
        graph.run({})
    
    # Test with missing required fields
    with pytest.raises(Exception):
        graph.run(GraphState(question_meta={}))

def test_graph_state_transitions(mock_user_profile):
    graph = create_cultural_graph()
    
    state = GraphState(
        question_meta={
            "original": "What are common wedding traditions?"
        },
        user_profile=mock_user_profile
    )
    
    result = graph.run(state)
    
    assert "current_state" in result
    assert result["current_state"] == "compose"
    assert "response_state" in result
    assert "expert_responses" in result["response_state"]

def test_database_interactions(mock_user_profile):
    graph = create_cultural_graph()
    
    state = GraphState(
        question_meta={
            "original": "How do different cultures celebrate holidays?",
            "sensitive_topics": ["cultural_practices"]
        },
        user_profile=mock_user_profile
    )
    
    # Test database write
    state.db_action = "write"
    state.db_key = "test_key"
    state.db_value = ["cultural_practices"]
    result = graph.run(state)
    assert "db_result" in result
    
    # Test database read
    state.db_action = "read"
    state.db_key = "test_key"
    result = graph.run(state)
    assert "db_result" in result
    assert result["db_result"] == ["cultural_practices"]

def test_edge_case_state_transitions():
    graph = create_cultural_graph()
    
    # Test missing user profile
    state = GraphState(
        question_meta={
            "original": "What are common wedding traditions?"
        }
    )
    
    with pytest.raises(Exception):
        graph.run(state)
    
    # Test invalid state transition
    with pytest.raises(Exception):
        invalid_state = GraphState(
            question_meta={
                "original": "Test question"
            },
            user_profile={
                "demographics": {"country": "US"},
                "preferences": {}
            },
            current_state="invalid_state"
        )
        graph.run(invalid_state)

def test_concurrent_expert_execution(mock_user_profile):
    graph = create_cultural_graph()
    
    state = GraphState(
        question_meta={
            "original": "How do different cultures celebrate new year?",
            "sensitive_topics": ["cultural_practices"],
            "relevant_cultures": ["US", "China", "India"]
        },
        user_profile=mock_user_profile
    )
    
    result = graph.run(state)
    
    assert "response_state" in result
    assert "expert_responses" in result["response_state"]
    responses = result["response_state"]["expert_responses"]
    cultures = [r["culture"] for r in responses]
    assert all(c in cultures for c in ["US", "China", "India"])