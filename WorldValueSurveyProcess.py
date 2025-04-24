# WorldValueSurveyProcess.py
# Embeds a persona dictionary into a vector using a BERT model

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import json
import os
from typing import Dict, List

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

def load_wvs_questions(filepath: str) -> Dict:
    """Load WVS questions from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate_persona_wvs(persona: dict, questions_data: Dict, output_dir: str = "artifacts") -> Dict:
    """Evaluate a persona against WVS questions and save results.
    
    Args:
        persona: Dictionary containing persona attributes
        questions_data: WVS questions data
        output_dir: Directory to save evaluation artifacts
    
    Returns:
        Dictionary containing evaluation results
    """
    # Import EvaluationLosses from evaluation.py
    from evaluation import EvaluationLosses

    try:
        os.makedirs(output_dir, exist_ok=True)
        persona_embedding = embed_persona(persona)
        results = {
            "persona": persona,
            "embedding": persona_embedding.tolist(),
            "questions": {}
        }

        # Example lambdas and label_map for demonstration
        lambdas = [1.0] * 7
        label_map = {"safe": 0, "unsafe": 1}  # Example; adjust as needed

        evaluator = EvaluationLosses(lambdas, label_map)

        for q_id, q_data in questions_data.items():
            # Example: Use the first question as the response
            response = q_data["questions"][0] if q_data["questions"] else ""
            # Dummy values for required fields
            topic_responses = [response]
            topics = ["safe"]  # Example topic
            cultural_ref = response
            style = persona_embedding  # Use embedding as dummy style
            same_culture_responses = [response]
            responseA = response
            responseB = response
            predictions = torch.tensor([[0.5, 0.5]])
            labels = torch.tensor([0])
            masks = torch.tensor([1])

            response_pack = {
                "response": response,
                "topic_responses": topic_responses,
                "topics": topics,
                "cultural_ref": cultural_ref,
                "style": torch.tensor(style),
                "same_culture_responses": same_culture_responses,
                "responseA": responseA,
                "responseB": responseB,
                "predictions": predictions,
                "labels": labels,
                "masks": masks
            }

            total_loss = evaluator.L_total(response_pack)
            results["questions"][q_id] = {
                "description": q_data["description"],
                "questions": q_data["questions"],
                "options": q_data["options"],
                "total_loss": float(total_loss.detach().cpu().numpy())
            }

        output_file = os.path.join(output_dir, "wvs_evaluation.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        return results

    except Exception as e:
        print(f"Error in WVS evaluation: {e}")
        raise

# Example usage:
if __name__ == "__main__":
    # Load WVS questions
    wvs_questions = load_wvs_questions("wvs_questions.json")
    
    # Example persona
    test_persona = {
        "name": "Test Person",
        "age": 30,
        "occupation": "Engineer",
        "values": "Family-oriented, hardworking"
    }
    
    # Run evaluation
    results = evaluate_persona_wvs(test_persona, wvs_questions)
