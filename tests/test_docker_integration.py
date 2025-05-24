"""
Integration tests for the full Docker Compose stack.
These tests verify that the Ollama service and application work together correctly.
"""
import pytest
import requests
import time
import subprocess
import os


class TestDockerIntegration:
    """Tests that require the full Docker Compose stack to be running."""
    
    @classmethod
    def setup_class(cls):
        """Start the Docker Compose stack before tests."""
        print("Starting Docker Compose stack...")
        # Change to project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(project_root)
        
        # Start services
        subprocess.run(["docker-compose", "up", "-d", "--build"], check=True)
        
        # Wait for services to be healthy
        max_retries = 30
        retry_delay = 2
        
        print("Waiting for services to be ready...")
        for i in range(max_retries):
            try:
                # Check Ollama service health
                ollama_health = subprocess.run(
                    ["docker-compose", "exec", "-T", "ollama-gpu", "curl", "-s", "http://localhost:11434/api/version"],
                    capture_output=True,
                    text=True
                )
                
                # Check application health
                app_response = requests.get("http://localhost:8000/", timeout=5)
                
                if ollama_health.returncode == 0 and app_response.status_code < 500:
                    print("Services are ready!")
                    break
            except (subprocess.CalledProcessError, requests.exceptions.RequestException):
                pass
            
            print(f"Waiting for services... ({i+1}/{max_retries})")
            time.sleep(retry_delay)
        else:
            raise RuntimeError("Services failed to start within timeout period")
    
    @classmethod
    def teardown_class(cls):
        """Stop the Docker Compose stack after tests."""
        print("Stopping Docker Compose stack...")
        subprocess.run(["docker-compose", "down", "-v"], check=True)
    
    def test_ollama_service_health(self):
        """Test that Ollama service is healthy and responding."""
        # Execute health check inside container
        result = subprocess.run(
            ["docker-compose", "exec", "-T", "ollama-gpu", "curl", "-s", "http://localhost:11434/api/version"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "version" in result.stdout.lower()
    
    def test_embed_endpoint(self):
        """Test the /api/embed endpoint with actual Ollama service."""
        response = requests.post(
            "http://localhost:8000/api/embed",
            json={"text": "Test embedding generation"},
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "embeddings" in data
        assert isinstance(data["embeddings"], list)
        assert len(data["embeddings"]) > 0
        assert all(isinstance(x, (int, float)) for x in data["embeddings"])
        
        # Verify embedding dimensions (mxbai-embed-large typically produces 1024-dim embeddings)
        assert len(data["embeddings"]) >= 512
    
    def test_generate_endpoint(self):
        """Test the /api/generate endpoint with actual Ollama service."""
        response = requests.post(
            "http://localhost:8000/api/generate",
            json={"prompt": "Hello, how are you?"},
            timeout=60
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "text" in data
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0
    
    def test_models_are_available(self):
        """Test that required models are pulled and available."""
        # Check phi4 model
        phi4_check = subprocess.run(
            ["docker-compose", "exec", "-T", "ollama-gpu", "ollama", "list"],
            capture_output=True,
            text=True
        )
        
        assert phi4_check.returncode == 0
        assert "phi4" in phi4_check.stdout
        assert "mxbai-embed-large" in phi4_check.stdout
    
    def test_concurrent_requests(self):
        """Test that the service can handle concurrent requests."""
        import concurrent.futures
        
        def make_embed_request(text):
            return requests.post(
                "http://localhost:8000/api/embed",
                json={"text": text},
                timeout=30
            )
        
        # Make 5 concurrent requests
        texts = [f"Test text {i}" for i in range(5)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_embed_request, text) for text in texts]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)
        assert all("embeddings" in r.json() for r in responses)


@pytest.mark.skipif(
    "SKIP_DOCKER_TESTS" in os.environ,
    reason="Skipping Docker integration tests"
)
class TestDockerComposeFixture:
    """Alternative integration tests using pytest-docker if available."""
    
    def test_compose_file_valid(self):
        """Test that docker-compose.yml is valid."""
        result = subprocess.run(
            ["docker-compose", "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "ollama-gpu" in result.stdout
        assert "cultural-agent" in result.stdout