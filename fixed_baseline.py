#!/usr/bin/env python3
"""
Fixed baseline that uses Ollama instead of Lambda API.
"""
from llmagentsetting.ollama_client import OllamaClient

def generate_baseline_essay_fixed(profiles: list, merged_question: str) -> str:
    """Generate baseline essay using Ollama client."""
    
    client = OllamaClient()
    user = profiles[0]

    prompt = f"""\
[System Instruction]
You are an AI language model designed to provide thoughtful, concise, and globally understandable answers. 
You must maintain cultural neutrality — avoid bias toward any specific country, religion, ethnicity, or tradition. 
Your tone should be respectful, balanced, and representative of a global average perspective.

[User Profile for Context Only]
{user}

[Task Instruction]
Please write a short essay (under 200 words) that responds to the following question. 
Your essay should be well-structured and objective. Do not mention specific cultural practices or identities.

[Question]
{merged_question}

[Essay]
"""
    
    try:
        response = client.generate(prompt)
        return response if response else "Unable to generate baseline response."
    except Exception as e:
        print(f"Baseline generation error: {e}")
        return "Error generating baseline response."

# Test the fixed version
if __name__ == "__main__":
    test_profile = [{"test": "profile"}]
    test_question = "What do you think about education?"
    
    result = generate_baseline_essay_fixed(test_profile, test_question)
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result)}")
    print(f"Result preview: {result[:200]}")