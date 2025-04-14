import json
import os
import pandas as pd
from src.graph import create_cultural_graph
from langgraph.checkpoint.memory import MemorySaver

def load_test_data():
    with open('data/test_questions.json', 'r') as file:
        data = json.load(file)
    
    rows = []
    for key, content in data.items():
        questions = content['questions']
        options = content['options']
        
        questions_str = str(questions).strip()[1:-1].strip()
        options_str = str(options).strip()[1:-1].strip()
        
        rows.append({
            'key': key,
            'description': content['description'],
            'questions': questions_str,
            'options': options_str
        })
    
    return pd.DataFrame(rows)

def test_cultural_graph_with_data():
    # Load test data
    df = load_test_data()
    
    # Create graph
    memory = MemorySaver()
    workflow = create_cultural_graph()
    debator = workflow.with_config(run_name="Testing Cultural Graph")
    
    results = []
    for _, row in df.iterrows():
        state = {
            "question_meta": {
                "original": row['questions']
            },
            "user_profile": {
                "id": "test_user",
                "demographics": {
                    "age": 30,
                    "location": "Global",
                    "gender": "neutral"
                },
                "preferences": {
                    "language": "English"
                }
            },
            "response_state": {
                "expert_responses": []
            },
            "full_history": []
        }
        
        # Invoke the graph
        try:
            result = debator.invoke(state)
            assert result is not None
            assert "response_state" in result
            assert "final" in result["response_state"]
            results.append({
                "question": row['questions'],
                "response": result["response_state"]["final"],
                "status": "success"
            })
        except Exception as e:
            results.append({
                "question": row['questions'],
                "error": str(e),
                "status": "failed"
            })
    
    # Verify results
    assert len(results) == len(df)
    assert all(r["status"] == "success" for r in results)

if __name__ == "__main__":
    test_cultural_graph_with_data()