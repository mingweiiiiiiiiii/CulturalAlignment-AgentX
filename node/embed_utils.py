import numpy as np
from typing import Dict, List, Any
import os
import requests
from typing import Dict, Any

TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", "http://129.146.102.0:8080/embed")
EMBED_DIM = 1024  # set this to whatever your model returns

def embed_persona_tei(persona: Dict[str, Any]) -> np.ndarray:
    text = ", ".join(f"{k}: {v}" for k, v in persona.items())
    payload = {"inputs": [text]}

    try:
        resp = requests.post(
            TEI_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"❌ TEI request failed: {e}")
        print(f"🔍 Text passed to TEI:\n{text}")
        return np.zeros(EMBED_DIM)

    # data might be a dict{"embeddings": [[...]]} or a raw list[[...]]
    if isinstance(data, dict) and "embeddings" in data:
        embeddings_list = data["embeddings"]
    elif isinstance(data, list):
        embeddings_list = data
    else:
        print(f"⚠️ Unexpected TEI response format: {data!r}")
        return np.zeros(EMBED_DIM)

    if not embeddings_list or not isinstance(embeddings_list[0], (list, tuple)):
        print(f"⚠️ No embeddings returned for text: {text}")
        return np.zeros(EMBED_DIM)

    # Take the first embedding vector
    vec = np.array(embeddings_list[0], dtype=float)
    if vec.shape[0] != EMBED_DIM:
        print(f"⚠️ Embedding dim mismatch (got {vec.shape[0]}, expected {EMBED_DIM})")
        # either pad or truncate
        out = np.zeros(EMBED_DIM)
        out[: min(vec.shape[0], EMBED_DIM)] = vec[:EMBED_DIM]
        return out

    return vec

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