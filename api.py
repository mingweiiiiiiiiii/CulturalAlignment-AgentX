from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import time
import ollama
import config
from node.embed_utils import get_embeddings
from node.sen_agent_node import generate_text

# Monkey patch to use Ollama
import llmagentsetting.llm_clients as llm_clients
from llmagentsetting.ollama_client import OllamaClient

class OllamaAPIClient(OllamaClient):
    def __init__(self, state=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state

llm_clients.LambdaAPIClient = OllamaAPIClient

from mylanggraph.graph import create_cultural_graph
from mylanggraph.types import GraphState

app = FastAPI(title="Cultural Alignment API")

# Initialize ollama client for cultural agent
ollama_client = ollama.Client(host=config.OLLAMA_HOST)

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embeddings: list[float]

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    response: str

@app.post("/api/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    embeddings = get_embeddings(request.text)
    return {"embeddings": embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings}

@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    text = generate_text(request.prompt)
    return {"response": text}

# Cultural Agent endpoint
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
        response = ollama_client.generate(
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
            embeddings = get_embeddings(embedding_text)
            if embeddings:
                result["embedding"] = embeddings
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Cultural Alignment API"}

# Full workflow endpoint
class WorkflowRequest(BaseModel):
    persona: Dict[str, str]
    question: str
    options: List[str]
    cultures: Optional[List[str]] = ["United States", "China", "India", "Japan", "Turkey", "Vietnam"]

class WorkflowResponse(BaseModel):
    is_sensitive: bool
    sensitivity_score: int
    sensitive_topics: List[str]
    relevant_cultures: List[str]
    expert_responses: List[Dict[str, Any]]
    final_response: str
    processing_time: float
    node_times: Dict[str, float]

@app.post("/api/cultural-workflow", response_model=WorkflowResponse)
def run_cultural_workflow(request: WorkflowRequest):
    """Run the full cultural sensitivity analysis workflow"""
    
    try:
        start_time = time.time()
        
        # Format question with options
        merged_question = f"{request.question}\n\nOptions:\n"
        for i, option in enumerate(request.options):
            merged_question += f"{chr(65 + i)}. {option}\n"
        
        # Create graph with specified cultures
        graph = create_cultural_graph(cultures=request.cultures)
        
        # Create initial state
        state: GraphState = {
            "user_profile": request.persona,
            "question_meta": {
                "original": merged_question,
                "options": request.options,
                "sensitive_topics": [],
                "relevant_cultures": [],
            },
            "response_state": {
                "expert_responses": [],
            },
            "full_history": [],
            "planner_counter": 0,
            "activate_sensitivity_check": True,
            "activate_extract_topics": True,
            "activate_router": False,
            "activate_judge": False,
            "activate_compose": False,
            "current_state": "planner",
            "node_times": {}
        }
        
        # Run the workflow
        result = graph.invoke(state, config={
            "recursion_limit": 50,
            "configurable": {"thread_id": f"api_{int(time.time())}"},
            "verbose": False,
        })
        
        elapsed_time = time.time() - start_time
        
        # Extract results
        meta = result.get("question_meta", {})
        response_state = result.get("response_state", {})
        
        return {
            "is_sensitive": meta.get("is_sensitive", False),
            "sensitivity_score": meta.get("sensitivity_score", 0),
            "sensitive_topics": [meta.get("sensitive_topics", "")] if isinstance(meta.get("sensitive_topics"), str) else meta.get("sensitive_topics", []),
            "relevant_cultures": meta.get("relevant_cultures", []),
            "expert_responses": response_state.get("expert_responses", []),
            "final_response": response_state.get("final", ""),
            "processing_time": elapsed_time,
            "node_times": result.get("node_times", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
