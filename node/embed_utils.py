import numpy as np
from typing import Dict, List, Any
import ollama
import config

# Initialize ollama client with host
client = ollama.Client(host=config.OLLAMA_HOST)

# Centralized embedding model name and options
EMBED_MODEL_NAME = "mxbai-embed-large"
EMBED_OPTIONS = {"num_ctx": 512}

def get_embeddings(text: str) -> List[float]:
    """Get embeddings for the given text using Ollama embed API."""
    response = client.embed(model=EMBED_MODEL_NAME, input=text, options=EMBED_OPTIONS)
    embeddings = response.get("embeddings", [])
    # Return the first embedding vector or empty list if none
    return embeddings[0] if embeddings else []

def embed_persona(persona: Dict[str, Any]) -> np.ndarray:
    text = ", ".join(f"{k}: {v}" for k, v in persona.items())
    embeddings_list = get_embeddings(text)
    if not embeddings_list:
        print(f"🔍 Text passed to embedding model:\n{text}")
        print(f"⚠️ No embeddings returned for text: {text}")
        return np.zeros(768)  # fallback: return zero vector of expected size

    return np.array(embeddings_list)

def embed_topics(state: Dict[str, Any]) -> List[np.ndarray]:
    """Embed a list of topics."""
    topics = state.get("topics", [])
    embeddings = []
    
    for topic in topics:
        emb = get_embeddings(topic)
        if emb:
            embeddings.append(np.array(emb))
        else:
            print(f"⚠️ No embeddings returned for topic: {topic}")
            embeddings.append(np.zeros(768))
    
    return embeddings