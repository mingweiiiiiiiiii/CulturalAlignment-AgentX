import pytest
import numpy as np
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# Patch embedding and generation functions for isolated endpoint tests
@pytest.fixture(autouse=True)
def patch_embed_generate(monkeypatch):
    monkeypatch.setattr(
        'node.embed_utils.get_embeddings',
        lambda text: np.ones(768)
    )
    monkeypatch.setattr(
        'node.sen_agent_node.generate_text',
        lambda prompt: "dummy response"
    )


def test_embed_endpoint_returns_embeddings_list():
    payload = {"text": "test input"}
    response = client.post("/api/embed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == 768
    # Values should be all ones
    assert all(val == 1.0 for val in data["embeddings"])


def test_generate_endpoint_returns_response_string():
    payload = {"prompt": "hello"}
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert data["response"] == "dummy response"
