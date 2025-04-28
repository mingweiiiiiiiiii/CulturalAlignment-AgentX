from typing import Optional, Dict, List, Any
from typing_extensions import TypedDict


# types.py
# Shared data structures used across the LangGraph pipeline



class UserProfile(TypedDict, total=False):
    id: str
    demographics: Dict[str, Any]
    preferences: Dict[str, Any]

class QuestionMeta(TypedDict, total=False):
    original: str
    index: Optional[int]
    is_sensitive: Optional[bool]
    sensitive_topics: List[str]
    relevant_cultures: List[str]

class ExpertResponse(TypedDict):
    culture: str
    response: str

class ResponseState(TypedDict, total=False):
    expert_responses: List[ExpertResponse]
    judged: Optional[str]
    final: Optional[str]

class GraphState(TypedDict, total=False):
    """A TypedDict class representing the state of the graph workflow.

    This class maintains various states and flags used throughout the graph processing pipeline.

    Attributes:
        question_meta (QuestionMeta): Metadata related to the current question.
        user_profile (UserProfile): User profile information.
        response_state (ResponseState): Current state of the response.
        full_history (List[str]): Complete history of interactions.
        planner_counter (int): Counter for planning iterations.
        activate_sensitivity_check (bool): Flag to enable sensitivity checking.
        activate_extract_topics (bool): Flag to enable topic extraction.
        activate_router (bool): Flag to enable routing functionality.
        activate_judge (bool): Flag to enable judging functionality.
        activate_compose (bool): Flag to enable composition functionality.
        db_action (Optional[str]): Database action to be performed.
        db_key (Optional[str]): Key for database operations.
        db_value (Optional[Any]): Value for database operations.
        db_result (Optional[Any]): Result from database operations.
        current_state (str): Current state in the workflow.
    """
    question_meta: QuestionMeta
    user_profile: UserProfile
    response_state: ResponseState
    full_history: List[str]

    planner_counter: int
    activate_sensitivity_check: bool
    activate_extract_topics: bool
    activate_router: bool
    activate_judge: bool
    activate_compose: bool

    db_action: Optional[str]
    db_key: Optional[str]
    db_value: Optional[Any]
    db_result: Optional[Any]

    current_state: str