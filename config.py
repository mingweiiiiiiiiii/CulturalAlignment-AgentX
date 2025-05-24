import os

# Load OLLAMA_HOST from environment, default to local host if not set
try:
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not OLLAMA_HOST:
        raise KeyError
except KeyError:
    raise EnvironmentError("Environment variable OLLAMA_HOST is required but was not set.")

# Optional API key for Ollama, leave as None if not provided
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", None)
