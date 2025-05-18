import numpy as np
from typing import Dict, List, Any
import ollama

def embed_persona(persona: Dict[str, Any]) -> np.ndarray:
    text = ", ".join(f"{k}: {v}" for k, v in persona.items())
    response = ollama.embed(model="mxbai-embed-large", input=text)
    embeddings = response.get("embeddings", [])

    if not embeddings:
        print(f"🔍 Text passed to embedding model:\n{text}")
        print(f"⚠️ No embeddings returned for text: {text}")
        return np.zeros(768)  # fallback: return zero vector of expected size

    return np.array(embeddings[0])