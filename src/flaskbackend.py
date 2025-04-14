from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import uvicorn
from langgraph.graph import StateGraph, END
from .types import GraphState
from .nodes import (
    planner_agent,
    determine_cultural_sensitivity,
    extract_sensitive_topics,
    route_to_cultures,
    judge_agent,
    compose_final_response,
    cultural_expert_node_factory,
    analyzer_router,
    CulturalExpert,
    USExpert,
    ChineseExpert,
    IndianExpert
)
from .db import database_node

# === FastAPI App Initialization ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Cultural Expert Registry ===
expert_classes = {
    "US": USExpert(),
    "China": ChineseExpert(),
    "India": IndianExpert()
}

# === Cultural Graph Builder ===
def create_cultural_graph(cultures: Optional[list[str]] = None):
    if cultures is None:
        cultures = ["US", "China", "India"]

    builder = StateGraph(GraphState)
    builder.set_entry_point("planner")

    builder.add_node("planner", planner_agent)
    builder.add_node("sensitivity_check", determine_cultural_sensitivity)
    builder.add_node("extract_topics", extract_sensitive_topics)
    builder.add_node("router", route_to_cultures)
    builder.add_node("compose", compose_final_response)
    builder.add_node("database", database_node)

    for culture, expert_instance in expert_classes.items():
        builder.add_node(f"expert_{culture}", expert_instance)

    builder.add_conditional_edges(
        "planner",
        lambda state: [state["__next__"]] if "__next__" in state else [],
        ["sensitivity_check", "router", "compose", "database"]
    )

    builder.add_edge("sensitivity_check", "extract_topics")
    builder.add_edge("extract_topics", "database")
    builder.add_conditional_edges("database", lambda state: ["planner"], ["planner"])
    builder.add_conditional_edges("router", analyzer_router, [f"expert_{c}" for c in expert_classes.keys()])

    for culture in expert_classes:
        builder.add_edge(f"expert_{culture}", "planner")

    builder.add_edge("compose", END)

    return builder.compile()

# === Instantiate the Graph ===
debate_graph = create_cultural_graph()

# === Endpoint for Triggering Workflow ===

@app.post("/trigger_workflow")
async def trigger_workflow(request: Request):
    data = await request.json()
    input_text = data.get("input", "What are gender roles in modern families?")  # Example sensitive question
    user_id = data.get("user_id", str(uuid4()))

    # Randomized example user profile
    user_profile: UserProfile = {
        "id": user_id,
        "demographics": {
            "age": random.randint(18, 65),
            "gender": random.choice(["male", "female", "non-binary"]),
            "location": random.choice(["US", "China", "India"])
        },
        "preferences": {
            "depth": random.choice(["brief", "detailed"]),
            "tone": random.choice(["formal", "casual"])
        }
    }

    # Constructing the initial state with all required flags and fields
    initial_state: GraphState = {
        "question_meta": {
            "original": input_text,
            "index": 0,
            "is_sensitive": None,
            "sensitive_topics": [],
            "relevant_cultures": []
        },
        "user_profile": user_profile,
        "response_state": {
            "expert_responses": [],
            "judged": None,
            "final": None
        },
        "full_history": [],
        "planner_counter": 0,
        "activate_sensitivity_check": True,
        "activate_extract_topics": True,
        "activate_router": True,
        "activate_judge": True,
        "activate_compose": True,
        "db_action": None,
        "db_key": None,
        "db_value": None,
        "db_result": None,
        "current_state": "init"
    }

    # Required thread config
    thread = {"configurable": {"thread_id": str(uuid4()), "recursion_limit": 100}}

    # Run the graph
    result = debate_graph.invoke(initial_state, thread)

    return {
        "input": input_text,
        "response": result
    }


# === Uvicorn Server (for local testing) ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
