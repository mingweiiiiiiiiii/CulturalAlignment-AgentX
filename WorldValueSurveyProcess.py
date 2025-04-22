# WorldValueSurveyProcess.py
# Embeds a persona dictionary into a vector using a BERT model

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed_persona(persona: dict) -> np.ndarray:
    # Validate persona input
    if not isinstance(persona, dict):
        raise TypeError("Persona must be a dictionary.")
    for k, v in persona.items():
        if not isinstance(k, str):
            raise TypeError("All persona keys must be strings.")
        if not isinstance(v, (str, int, float)):
            raise TypeError("All persona values must be str, int, or float.")
    try:
        text = ", ".join(f"{k}: {v}" for k, v in persona.items())
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        embedding_np = embeddings.squeeze().numpy()
        # Validate output embedding
        if embedding_np.ndim != 1 or embedding_np.shape[0] != 768:
            raise ValueError(f"Output embedding must be 1D with dimension 768, got shape {embedding_np.shape}")
        return embedding_np
    except Exception as e:
        print(f"Error embedding persona: {e}")
        raise
