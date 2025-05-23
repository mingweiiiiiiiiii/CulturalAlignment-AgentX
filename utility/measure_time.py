import time

def measure_time(node_function):
    """Decorator to measure and log the execution time of a node function."""
    def wrapper(state, *args, **kwargs):
        start_time = time.time()
        #print(f"Starting node: {node_function.__name__}")
        # Initialize API calls counter in state
        if "api_calls" not in state:
            state["api_calls"] = {}
        state["current_node"] = node_function.__name__
        
        result = node_function(state, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Node {node_function.__name__} completed in {elapsed_time:.2f} seconds.\n")

        # Store timing data in state for later analysis
        if "node_times" not in state:
            state["node_times"] = {}
        state["node_times"][node_function.__name__] = elapsed_time

        return result
    return wrapper