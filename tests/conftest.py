import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_embedding():
    return np.random.rand(768)

@pytest.fixture
def mock_user_profile():
    return {
        "demographics": {
            "country": "US",
            "age": 30,
            "sex": "Female",
            "education": "Bachelor's degree"
        },
        "preferences": {}
    }

@pytest.fixture
def mock_graph_state(mock_user_profile, mock_embedding):
    return {
        "question_meta": {
            "original": "How do different cultures celebrate holidays?",
            "sensitive_topics": ["cultural_practices"]
        },
        "user_profile": mock_user_profile,
        "user_embedding": mock_embedding
    }