from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import time
import json

# Monkey patch to use Ollama
import llmagentsetting.llm_clients as llm_clients
from llmagentsetting.ollama_client import OllamaClient

class OllamaAPIClient(OllamaClient):
    def __init__(self, state=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state

llm_clients.LambdaAPIClient = OllamaAPIClient

from mylanggraph.graph_enhanced import create_enhanced_cultural_graph
from mylanggraph.types import GraphState
from utility.cache_manager import get_cache_manager

app = FastAPI(title="Enhanced Cultural Alignment API")

# Pre-initialize enhanced graph at startup
print("Pre-initializing enhanced graph...")
_cached_enhanced_graph = None

def get_enhanced_graph():
    global _cached_enhanced_graph
    if _cached_enhanced_graph is None:
        _cached_enhanced_graph = create_enhanced_cultural_graph()
    return _cached_enhanced_graph

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    get_enhanced_graph()  # Pre-initialize
    print("✅ Enhanced graph initialized")

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
    cache_stats: Dict[str, Any]
    optimization_metrics: Dict[str, Any]
    enhancement_metrics: Dict[str, Any]

@app.post("/api/cultural-workflow-enhanced", response_model=WorkflowResponse)
def run_enhanced_workflow(request: WorkflowRequest):
    """Run the enhanced cultural sensitivity analysis workflow with improved detection"""
    
    try:
        start_time = time.time()
        
        # Format question with options
        merged_question = f"{request.question}\n\nOptions:\n"
        for i, option in enumerate(request.options):
            merged_question += f"{chr(65 + i)}. {option}\n"
        
        # Get pre-initialized enhanced graph
        graph = get_enhanced_graph()
        
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
            "current_state": "planner",
            "node_times": {}
        }
        
        # Run the enhanced workflow
        result = graph.invoke(state, config={
            "recursion_limit": 50,
            "configurable": {"thread_id": f"api_enh_{int(time.time())}"},
            "verbose": False,
        })
        
        elapsed_time = time.time() - start_time
        
        # Get cache statistics
        cache = get_cache_manager()
        cache_stats = cache.get_cache_stats()
        
        # Extract results
        meta = result.get("question_meta", {})
        response_state = result.get("response_state", {})
        
        # Calculate optimization metrics
        parallel_time = response_state.get("parallel_time", 0)
        sequential_time = len(response_state.get("expert_responses", [])) * 20  # Estimated
        time_saved = max(0, sequential_time - parallel_time)
        
        optimization_metrics = {
            "parallel_execution_time": parallel_time,
            "estimated_sequential_time": sequential_time,
            "time_saved": time_saved,
            "speedup_factor": sequential_time / parallel_time if parallel_time > 0 else 1,
            "total_llm_calls": result.get("planner_counter", 0) + len(response_state.get("expert_responses", [])) + 2
        }
        
        # Enhancement-specific metrics
        enhancement_metrics = {
            "enhanced_sensitivity_used": True,
            "threshold_used": "4/10 (lowered from 5/10)",
            "topic_boosts_applied": True,
            "cultural_context_considered": True,
            "boost_details": "Education +2, Hierarchy +2, Work +1, Cultural +1"
        }
        
        return {
            "is_sensitive": meta.get("is_sensitive", False),
            "sensitivity_score": meta.get("sensitivity_score", 0),
            "sensitive_topics": [meta.get("sensitive_topics", "")] if isinstance(meta.get("sensitive_topics"), str) else meta.get("sensitive_topics", []),
            "relevant_cultures": meta.get("relevant_cultures", []),
            "expert_responses": response_state.get("expert_responses", []),
            "final_response": response_state.get("final", ""),
            "processing_time": elapsed_time,
            "node_times": result.get("node_times", {}),
            "cache_stats": cache_stats,
            "optimization_metrics": optimization_metrics,
            "enhancement_metrics": enhancement_metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Enhanced Cultural Alignment API"}

@app.get("/cache-stats")
def get_cache_statistics():
    """Get current cache statistics"""
    cache = get_cache_manager()
    return cache.get_cache_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)