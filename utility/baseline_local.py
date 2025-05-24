"""
Local baseline essay generator using ollama.
"""
from llmagentsetting.ollama_client import OllamaClient

def generate_baseline_essay(profiles: list, merged_question: str) -> str:
    """
    Generate a baseline essay that maintains cultural neutrality.
    """
    client = OllamaClient()
    
    # Extract user info if provided
    user = profiles[0] if profiles else {}
    
    prompt = f"""You are an AI language model designed to provide thoughtful, concise, and globally understandable answers. 
You must maintain cultural neutrality — avoid bias toward any specific country, religion, ethnicity, or tradition. 
Your tone should be respectful, balanced, and representative of a global average perspective.

Please write a short essay (under 200 words) that responds to the following question. 
Your essay should be well-structured and objective. Do not mention specific cultural practices or identities.

Question: {merged_question}

Essay:"""
    
    try:
        response = client.generate(prompt)
        return response
    except Exception as e:
        return f"Error generating baseline essay: {str(e)}"