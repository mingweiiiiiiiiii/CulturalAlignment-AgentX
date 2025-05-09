#!/usr/bin/env sh
# entrypoint.sh: Start Ollama daemon, pull model, then run the application

# Start Ollama API server in the background
ollama serve &

# Wait for the server to be ready
sleep 5

# Pull the embedding model
ollama pull mxbai-embed-large

# Execute the main application
exec python main.py
