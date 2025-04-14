from typing import Dict
from .types import GraphState

# Simple in-memory database
simple_db = {}

def database_node(state: GraphState) -> Dict:
    """Performs database operations based on the provided GraphState.

    This function handles read and write operations on a simple key-value database store.
    It processes the state's db_action to either write a value to the database or read
    a value from it.

    Args:
        state (GraphState): A state object containing database operation parameters:
            - db_action: String indicating the operation ('read' or 'write')
            - db_key: The key to read from or write to
            - db_value: The value to write (only used for 'write' operations)

    Returns:
        Dict: A dictionary containing:
            - db_result: The result of the read operation (None for write operations)
            - current_state: String indicating the current state ('database')

    Examples:
        >>> state = GraphState(db_action='write', db_key='test', db_value='data')
        >>> database_node(state)
        {'db_result': None, 'current_state': 'database'}

        >>> state = GraphState(db_action='read', db_key='test')
        >>> database_node(state)
        {'db_result': 'data', 'current_state': 'database'}
    """
    action = state.get("db_action")
    key = state.get("db_key")
    value = state.get("db_value")
    result = None

    if action == "write" and key:
        simple_db[key] = value
    elif action == "read" and key:
        result = simple_db.get(key)

    return {
        "db_result": result,
        "current_state": "database"
    }