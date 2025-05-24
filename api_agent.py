from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import ollama
import config

app = FastAPI(title="Cultural Agent API")

# Initialize ollama client
client = ollama.Client(host=config.OLLAMA_HOST)

class AgentRequest(BaseModel):
    persona: Dict[str, str]
    question: str
    options: List[str]
    include_embeddings: bool = False

class AgentResponse(BaseModel):
    response: str
    embedding: Optional[List[float]] = None
    persona_summary: str

@app.post("/api/cultural-agent", response_model=AgentResponse)
def run_cultural_agent(request: AgentRequest):
    """Run the cultural agent with a given persona and question"""
    
    try:
        # Format the question with options
        formatted_question = f"{request.question}\n\nOptions:\n"
        for i, option in enumerate(request.options):
            formatted_question += f"{chr(65 + i)}. {option}\n"
        
        # Create prompt that incorporates cultural perspective
        prompt = f"""You are responding to a survey question as someone with the following background:
{json.dumps(request.persona, indent=2)}

Please respond to this question in a way that reflects your cultural background and personal experiences:

{formatted_question}

Provide a thoughtful response that considers your cultural identity, values, and life experiences. 
Be authentic to the persona while answering the question."""

        # Generate response using Ollama
        response = client.generate(
            model="phi4",
            prompt=prompt,
            options={"temperature": 0.7, "max_tokens": 500}
        )
        
        # Extract response text
        if hasattr(response, 'response'):
            answer = response.response
        else:
            answer = str(response)
        
        # Create persona summary
        persona_summary = f"{request.persona.get('age', 'Unknown')} year old {request.persona.get('race', 'Unknown')} {request.persona.get('sex', 'person')} from {request.persona.get('ancestry', 'Unknown')} background"
        
        result = {
            "response": answer,
            "persona_summary": persona_summary
        }
        
        # Generate embeddings if requested
        if request.include_embeddings:
            embedding_text = " ".join([
                str(v) for v in request.persona.values() if isinstance(v, str)
            ])
            
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
                
            if embeddings:
                result["embedding"] = embeddings[0]
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Cultural Agent API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)