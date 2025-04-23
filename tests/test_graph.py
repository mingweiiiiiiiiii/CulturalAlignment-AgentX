import pytest
from ..graph import create_cultural_graph, GraphState

def test_graph_creation():
    # Test with default cultures
    graph = create_cultural_graph()
    assert graph is not None

    # Test with custom cultures
    custom_cultures = ["US", "Japan", "India"]
    graph = create_cultural_graph(custom_cultures)
    assert graph is not None

def test_graph_execution():
    graph = create_cultural_graph()
    
    # Test simple state
    initial_state = GraphState(
        question_meta={
            "original": "How do different cultures celebrate holidays?"
        },
        user_profile={
            "demographics": {
                "country": "US",
                "age": 30
            },
            "preferences": {}
        }
    )
    
    result = graph.run(initial_state)
    assert result is not None
    assert "response_state" in result
    assert "final" in result["response_state"]

def test_graph_error_handling():
    graph = create_cultural_graph()
    
    # Test with invalid state
    with pytest.raises(Exception):
        graph.run({})  # Empty state should raise error
    
    # Test with missing required fields
    with pytest.raises(Exception):
        graph.run({"question_meta": {}})  # Missing original question

def test_graph_state_transitions():
    graph = create_cultural_graph()
    
    state = GraphState(
        question_meta={
            "original": "What are common wedding traditions?"
        },
        user_profile={
            "demographics": {
                "country": "US",
                "age": 25
            },
            "preferences": {}
        }
    )
    
    result = graph.run(state)
    
    # Verify state transitions
    assert "current_state" in result
    assert result["current_state"] == "compose"  # Final state should be compose
    assert "response_state" in result
    assert "expert_responses" in result["response_state"]

def test_database_interactions():
    graph = create_cultural_graph()
    
    state = GraphState(
        question_meta={
            "original": "How do different cultures celebrate holidays?",
            "sensitive_topics": ["cultural_practices"]
        },
        user_profile={
            "demographics": {"country": "US"},
            "preferences": {}
        }
    )
    
    # Test database write
    state["db_action"] = "write"
    state["db_key"] = "test_key"
    state["db_value"] = ["cultural_practices"]
    result = graph.run(state)
    assert "db_result" in result
    
    # Test database read
    state["db_action"] = "read"
    state["db_key"] = "test_key"
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
    state = GraphState(
        question_meta={
            "original": "Test question"
        },
        user_profile={
            "demographics": {"country": "US"},
            "preferences": {}
        },
        current_state="invalid_state"
    )
    
    with pytest.raises(Exception):
        graph.run(state)

def test_concurrent_expert_execution():
    graph = create_cultural_graph()
    
    state = GraphState(
        question_meta={
            "original": "How do different cultures celebrate new year?",
            "sensitive_topics": ["cultural_practices"],
            "relevant_cultures": ["US", "China", "India"]
        },
        user_profile={
            "demographics": {"country": "US"},
            "preferences": {}
        }
    )
    
    result = graph.run(state)
    
    # Verify all experts were executed
    assert "response_state" in result
    assert "expert_responses" in result["response_state"]
    responses = result["response_state"]["expert_responses"]
    cultures = [r["culture"] for r in responses]
    assert all(c in cultures for c in ["US", "China", "India"])