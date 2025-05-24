#!/usr/bin/env python3
"""
Simple test to demonstrate the cultural agent using Ollama
"""
import json
import ollama
import config

# Initialize ollama client
client = ollama.Client(host=config.OLLAMA_HOST)

def test_cultural_response():
    """Test cultural agent response using Ollama directly"""
    
    # Sample persona
    persona = {
        "age": "22",
        "sex": "Male", 
        "race": "Two or More Races",
        "ancestry": "Puerto Rican",
        "household language": "Spanish",
        "lifestyle": "Eco-conscious and community-oriented",
        "ideology": "Progressive",
        "religion": "Catholic"
    }
    
    # Sample question
    question = """During the past year, how frequently have you or your family experienced a lack of secure housing?

Options:
A. Often
B. Sometimes  
C. Rarely
D. Never
E. Don't know"""

    # Create prompt that incorporates cultural perspective
    prompt = f"""You are responding to a survey question as someone with the following background:
{json.dumps(persona, indent=2)}

Please respond to this question in a way that reflects your cultural background and personal experiences:

{question}

Provide a thoughtful response that considers your Puerto Rican heritage, Spanish-speaking household, progressive values, and eco-conscious lifestyle."""

    print("=== Testing Cultural Agent with Ollama ===")
    print(f"\nPersona: {json.dumps(persona, indent=2)}")
    print(f"\nQuestion: {question}")
    print("\n--- Generating Response ---")
    
    try:
        # Generate response using Ollama
        response = client.generate(
            model="phi4",
            prompt=prompt,
            options={"temperature": 0.7, "max_tokens": 500}
        )
        
        # Extract the response text
        if hasattr(response, 'response'):
            answer = response.response
        else:
            answer = str(response)
            
        print(f"\nCultural Agent Response:\n{answer}")
        
        # Test embedding functionality
        print("\n--- Testing Embedding Functionality ---")
        embedding_text = f"{persona['ancestry']} {persona['lifestyle']} {persona['ideology']}"
        embedding_response = client.embed(
            model="mxbai-embed-large",
            input=embedding_text
        )
        
        if hasattr(embedding_response, 'embeddings'):
            embeddings = embedding_response.embeddings
        elif isinstance(embedding_response, dict) and 'embeddings' in embedding_response:
            embeddings = embedding_response['embeddings']
        else:
            embeddings = []
            
        print(f"Generated embedding vector of length: {len(embeddings[0]) if embeddings else 0}")
        
        return answer
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_cultural_response()