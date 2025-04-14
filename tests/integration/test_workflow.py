import pytest
from src.graph import create_cultural_graph
from src.types import GraphState

@pytest.fixture
def workflow():
    return create_cultural_graph(cultures=["TestCulture1", "TestCulture2"])

@pytest.fixture
def sample_input_state():
    return {
        "question_meta": {
            "original": "How do different cultures view personal space?"
        },
        "user_profile": {
            "id": "test_user_integration",
            "demographics": {
                "region": "Global",
                "age": 25
            },
            "preferences": {
                "language": "English"
            }
        },
        "response_state": {
            "expert_responses": []
        },
        "full_history": []
    }

def test_complete_workflow(workflow, sample_input_state):
    """Test the complete workflow from question to final response"""
    result = workflow.invoke(sample_input_state)
    
    # Verify the workflow completed successfully
    assert result is not None
    assert "response_state" in result
    
    # Check if we have expert responses
    assert "expert_responses" in result["response_state"]
    assert len(result["response_state"]["expert_responses"]) > 0
    
    # Verify we have a final response
    assert "final" in result["response_state"]
    assert isinstance(result["response_state"]["final"], str)
    
    # Check if sensitive topics were identified
    assert "question_meta" in result
    assert "sensitive_topics" in result["question_meta"]
    
def test_workflow_error_handling(workflow):
    """Test workflow behavior with invalid input"""
    invalid_state = {
        "question_meta": {
            "original": ""  # Empty question
        }
    }
    
    with pytest.raises(Exception):
        workflow.invoke(invalid_state)

@pytest.mark.asyncio
async def test_concurrent_workflow_execution(workflow, sample_input_state):
    """Test multiple concurrent workflow executions"""
    import asyncio
    
    async def run_workflow():
        return workflow.invoke(sample_input_state)
    
    # Run multiple workflows concurrently
    tasks = [run_workflow() for _ in range(3)]
    results = await asyncio.gather(*tasks)
    
    # Verify all executions completed successfully
    for result in results:
        assert result is not None
        assert "response_state" in result
        assert "final" in result["response_state"]