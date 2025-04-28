import pytest
import numpy as np
from ..inputData import PersonaSampler
from ..nodes import (
    determine_cultural_sensitivity,
    extract_sensitive_topics,
    route_to_cultures,
)
from ..graph import create_cultural_graph, GraphState
from ..WorldValueSurveyProcess import embed_persona
from ..evaluation import EvaluationLosses


def test_end_to_end_flow():
    """Test the complete flow from user input to final response"""
    # Initialize components
    sampler = PersonaSampler()
    graph = create_cultural_graph()

    # Generate test profile and question
    profile = sampler.sample_profiles(n=1)[0]
    question, options = sampler.sample_question()

    # Create initial state
    initial_state = GraphState(
        question_meta={"original": question, "options": options},
        user_profile=profile,
        user_embedding=embed_persona(profile),
    )

    # Run through graph
    final_state = graph.run(initial_state)

    # Verify state transitions and outputs
    assert "response_state" in final_state
    assert "final" in final_state["response_state"]
    assert isinstance(final_state["response_state"]["final"], str)


def test_cultural_sensitivity_to_routing():
    """Test the flow from sensitivity detection to cultural routing"""
    # Create test state
    state = GraphState(
        question_meta={
            "original": "How do different cultures view marriage traditions?"
        },
        user_profile={"demographics": {"country": "US"}, "preferences": {}},
    )

    # Run sensitivity detection
    sensitivity_result = determine_cultural_sensitivity(state)
    assert "is_sensitive" in sensitivity_result["question_meta"]

    # Extract topics
    topics_result = extract_sensitive_topics(sensitivity_result)
    assert "sensitive_topics" in topics_result["question_meta"]

    # Route to cultures
    routing_result = route_to_cultures(
        topics_result, ["US", "China", "India"], np.random.rand(3, 768)
    )
    assert "relevant_cultures" in routing_result["question_meta"]


def test_evaluation_pipeline():
    """Test the evaluation metrics pipeline"""
    # Initialize components
    sampler = PersonaSampler()
    graph = create_cultural_graph()
    evaluator = EvaluationLosses(lambdas=[1.0] * 7, label_map={"safe": 0, "unsafe": 1})

    # Generate test data
    profile = sampler.sample_profiles(n=1)[0]
    question, _ = sampler.sample_question()

    # Run through graph
    state = GraphState(
        question_meta={"original": question},
        user_profile=profile,
        user_embedding=embed_persona(profile),
    )
    result = graph.run(state)

    # Evaluate response
    response_pack = {
        "response": result["response_state"]["final"],
        "topic_responses": result["response_state"].get("expert_responses", []),
        "topics": [question],
        "cultural_ref": "Cultural reference",
        "style": "Target style",
        "same_culture_responses": result["response_state"].get("expert_responses", []),
        "responseA": "Response A",
        "responseB": "Response B",
        "predictions": [],
        "labels": [],
        "masks": [],
    }

    total_loss = evaluator.L_total(response_pack)
    assert isinstance(total_loss, float)


def test_error_propagation():
    """Test how errors propagate through the system"""
    graph = create_cultural_graph()

    # Test with invalid state
    with pytest.raises(Exception):
        graph.run({})

    # Test with missing required fields
    with pytest.raises(Exception):
        graph.run(GraphState(question_meta={"original": ""}))

    # Test with invalid culture
    with pytest.raises(Exception):
        graph.run(
            GraphState(
                question_meta={
                    "original": "Test question",
                    "relevant_cultures": ["InvalidCulture"],
                },
                user_profile={"demographics": {"country": "US"}},
            )
        )


def test_concurrent_processing():
    """Test concurrent processing of cultural experts"""
    graph = create_cultural_graph()
    sampler = PersonaSampler()

    # Generate multiple test cases
    profiles = sampler.sample_profiles(n=3)
    questions = [sampler.sample_question()[0] for _ in range(3)]

    # Process concurrently
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for profile, question in zip(profiles, questions):
            state = GraphState(
                question_meta={"original": question},
                user_profile=profile,
                user_embedding=embed_persona(profile),
            )
            futures.append(executor.submit(graph.run, state))

        results = [f.result() for f in futures]

    # Verify results
    assert len(results) == 3
    for result in results:
        assert "response_state" in result
        assert "final" in result["response_state"]


import pytest
from graph import create_cultural_graph, GraphState
from nodes import (
    determine_cultural_sensitivity,
    extract_sensitive_topics,
    route_to_cultures,
)
from inputData import process_user_input


def test_full_conversation_flow(mock_user_profile, mock_embedding):
    # Initialize graph
    graph = create_cultural_graph()

    # Test simple cultural question
    input_text = "How do different cultures celebrate New Year?"
    initial_state = process_user_input(input_text, mock_user_profile)

    # Process through graph
    result = graph.run(GraphState(**initial_state))

    assert result is not None
    assert "response_state" in result
    assert "final" in result["response_state"]
    assert isinstance(result["response_state"]["final"], str)
    assert len(result["response_state"]["final"]) > 0


def test_sensitive_topic_handling(mock_user_profile):
    graph = create_cultural_graph()

    # Test sensitive cultural question
    input_text = "Why do some cultures have controversial traditions?"
    initial_state = process_user_input(input_text, mock_user_profile)

    # Process through sensitivity detection
    state = determine_cultural_sensitivity(initial_state)
    assert state["question_meta"]["is_sensitive"] == True

    # Extract topics
    state = extract_sensitive_topics(state)
    assert "sensitive_topics" in state["question_meta"]
    assert len(state["question_meta"]["sensitive_topics"]) > 0

    # Full graph processing
    result = graph.run(GraphState(**initial_state))
    assert "response_state" in result
    assert "final" in result["response_state"]

    # Check for sensitivity acknowledgment in response
    response = result["response_state"]["final"].lower()
    assert any(
        word in response
        for word in ["respect", "sensitive", "understand", "perspective"]
    )


def test_multi_cultural_routing(mock_user_profile, mock_embedding):
    graph = create_cultural_graph()

    # Test question involving multiple cultures
    input_text = "Compare wedding traditions in different cultures"
    initial_state = process_user_input(input_text, mock_user_profile)

    # Process cultural routing
    culture_embeddings = {
        "US": mock_embedding,
        "China": mock_embedding,
        "India": mock_embedding,
    }

    state = route_to_cultures(
        initial_state,
        list(culture_embeddings.keys()),
        list(culture_embeddings.values()),
    )

    assert "relevant_cultures" in state["question_meta"]
    assert len(state["question_meta"]["relevant_cultures"]) > 1

    # Full graph processing
    result = graph.run(GraphState(**initial_state))
    assert "response_state" in result
    assert "expert_responses" in result["response_state"]

    # Verify multiple cultural perspectives
    expert_responses = result["response_state"]["expert_responses"]
    cultures_represented = {resp["culture"] for resp in expert_responses}
    assert len(cultures_represented) > 1


def test_error_handling_integration(mock_user_profile):
    graph = create_cultural_graph()

    # Test with invalid input
    with pytest.raises(Exception):
        graph.run(GraphState({}))

    # Test with missing user profile
    with pytest.raises(Exception):
        graph.run(GraphState(question_meta={"original": "test"}))

    # Test with invalid state transitions
    with pytest.raises(Exception):
        invalid_state = GraphState(
            question_meta={"original": "test"},
            user_profile=mock_user_profile,
            current_state="invalid_state",
        )
        graph.run(invalid_state)


def test_cultural_alignment_flow(mock_user_profile, mock_embedding):
    graph = create_cultural_graph()

    # Test cultural alignment with user profile
    input_text = "What are appropriate business meeting customs?"
    initial_state = process_user_input(input_text, mock_user_profile)

    result = graph.run(GraphState(**initial_state))

    # Verify response aligns with user's cultural background
    user_country = mock_user_profile["demographics"]["country"]
    response = result["response_state"]["final"]

    # Check if the response prioritizes user's cultural context
    assert any(
        expert["culture"] == user_country
        for expert in result["response_state"]["expert_responses"]
    )

    # Verify cultural sensitivity
    assert "sensitivity_score" in result["question_meta"]
    assert 0 <= result["question_meta"]["sensitivity_score"] <= 10


def test_cross_cultural_comparison(mock_user_profile):
    graph = create_cultural_graph()

    # Test explicit cross-cultural comparison
    input_text = "Compare dining etiquette in US, China, and India"
    initial_state = process_user_input(input_text, mock_user_profile)

    result = graph.run(GraphState(**initial_state))

    # Verify multiple cultural perspectives are included
    expert_responses = result["response_state"]["expert_responses"]
    cultures = {resp["culture"] for resp in expert_responses}

    assert "US" in cultures
    assert "China" in cultures
    assert "India" in cultures

    # Verify balanced representation
    response_lengths = {
        culture: len(resp["response"])
        for resp, culture in zip(expert_responses, cultures)
    }

    # Check if responses are roughly balanced in length
    lengths = list(response_lengths.values())
    max_length = max(lengths)
    min_length = min(lengths)
    assert min_length > 0
    assert max_length / min_length < 3  # No response should be 3x longer than others
