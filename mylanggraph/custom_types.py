from typing import Optional, Dict, List, Any

# mylanggraph/custom_types.py
from typing import TypedDict, Optional, List, Dict, Any

class GraphState(TypedDict, total=False):
    """
    Schema used for LangGraph state propagation
    """
    question_meta: Dict[str, Any]
    user_profile: Optional[Dict[str, Any]]
    user_embedding: Optional[List[float]]
    current_state: str
    response_state: Dict[str, Any]
    db_action: Optional[str]
    db_key: Optional[str]
    db_value: Optional[Any]
    __next__: str  # required for conditional branching

    # Add these keys used in your planner_agent
    planner_counter: int
    is_sensitive: Optional[bool]
    activate_sensitivity_check: Optional[bool]
    activate_extract_topics: Optional[bool]
    activate_router: Optional[bool]
    topics_extracted: Optional[bool]
    
    activate_set: List[Dict[str, Any]]
    node_times: Dict[str, float]
    api_calls: Dict[str, int]
"""
class GraphState:
    
    #State object for managing conversation flow through the graph
    

    def __init__(
        self,
        question_meta: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None,
        user_embedding: Optional[List[float]] = None,
        current_state: str = "init",
        response_state: Optional[Dict[str, Any]] = None,
        db_action: Optional[str] = None,
        db_key: Optional[str] = None,
        db_value: Optional[Any] = None,
    ):
        self.question_meta = question_meta
        self.user_profile = user_profile
        self.user_embedding = user_embedding
        self.current_state = current_state
        self.response_state = response_state or {}
        self.db_action = db_action
        self.db_key = db_key
        self.db_value = db_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_meta": self.question_meta,
            "user_profile": self.user_profile,
            "user_embedding": self.user_embedding,
            "current_state": self.current_state,
            "response_state": self.response_state,
            "db_action": self.db_action,
            "db_key": self.db_key,
            "db_value": self.db_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphState":
        return cls(**data)"""
