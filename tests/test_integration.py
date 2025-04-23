import pytest
import numpy as np
from ..inputData import PersonaSampler
from ..nodes import determine_cultural_sensitivity, extract_sensitive_topics, route_to_cultures
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
        question_meta={
            "original": question,
            "options": options
        },
        user_profile=profile,
        user_embedding=embed_persona(profile)
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
        user_profile={
            "demographics": {"country": "US"},
            "preferences": {}
        }
    )
    
    # Run sensitivity detection
    sensitivity_result = determine_cultural_sensitivity(state)
    assert "is_sensitive" in sensitivity_result["question_meta"]
    
    # Extract topics
    topics_result = extract_sensitive_topics(sensitivity_result)
    assert "sensitive_topics" in topics_result["question_meta"]
    
    # Route to cultures
    routing_result = route_to_cultures(
        topics_result,
        ["US", "China", "India"],
        np.random.rand(3, 768)
    )
    assert "relevant_cultures" in routing_result["question_meta"]

def test_evaluation_pipeline():
    """Test the evaluation metrics pipeline"""
    # Initialize components
    sampler = PersonaSampler()
    graph = create_cultural_graph()
    evaluator = EvaluationLosses(
        lambdas=[1.0] * 7,
        label_map={"safe": 0, "unsafe": 1}
    )
    
    # Generate test data
    profile = sampler.sample_profiles(n=1)[0]
    question, _ = sampler.sample_question()
    
    # Run through graph
    state = GraphState(
        question_meta={"original": question},
        user_profile=profile,
        user_embedding=embed_persona(profile)
    )
    result = graph.run(state)
    
    # Evaluate response
    response_pack = {
        'response': result['response_state']['final'],
        'topic_responses': result['response_state'].get('expert_responses', []),
        'topics': [question],
        'cultural_ref': "Cultural reference",
        'style': "Target style",
        'same_culture_responses': result['response_state'].get('expert_responses', []),
        'responseA': "Response A",
        'responseB': "Response B",
        'predictions': [],
        'labels': [],
        'masks': []
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
        graph.run(GraphState(
            question_meta={"original": ""}
        ))
    
    # Test with invalid culture
    with pytest.raises(Exception):
        graph.run(GraphState(
            question_meta={
                "original": "Test question",
                "relevant_cultures": ["InvalidCulture"]
            },
            user_profile={"demographics": {"country": "US"}}
        ))

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
                user_embedding=embed_persona(profile)
            )
            futures.append(executor.submit(graph.run, state))
        
        results = [f.result() for f in futures]
    
    # Verify results
    assert len(results) == 3
    for result in results:
        assert "response_state" in result
        assert "final" in result["response_state"]