# Cultural Alignment AgentX

A sophisticated AI system designed to provide culturally-aware and sensitive responses by consulting multiple cultural perspectives using LangGraph and LangChain.

## Overview

Cultural Alignment AgentX is an advanced question-answering system that:
- Analyzes questions for cultural sensitivity
- Identifies culturally sensitive topics
- Routes questions to culture-specific expert agents
- Aggregates responses from multiple cultural perspectives
- Provides balanced, culturally-informed answers

## Key Features

- **Cultural Sensitivity Detection**: Automatically identifies if a question touches on culturally sensitive topics
- **Multi-Cultural Consultation**: Dynamically routes questions to relevant cultural expert agents
- **Intelligent Response Aggregation**: Combines perspectives from multiple cultures into cohesive answers
- **State Management**: Uses LangGraph for sophisticated state handling and workflow management
- **Extensible Architecture**: Easy to add new cultural experts and modify the processing pipeline

## Prerequisites

- Python 3.9 or higher
- Groq API key for LLM access
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/CulturalAlignment-AgentX.git
cd CulturalAlignment-AgentX
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Set up your Groq API key:
   - For local development:
     ```python
     import os
     os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
     ```
   - For GitHub Actions:
     1. Go to your repository Settings > Secrets and Variables > Actions
     2. Add a new secret with name `GROQ_API_KEY` and your API key as value
     3. The workflows will automatically use this secret for testing

## Project Structure

```
CulturalAlignment-AgentX/
├── src/
│   ├── types.py        # Type definitions and state structures
│   ├── nodes.py        # Agent implementations
│   ├── graph.py        # Graph construction and workflow
│   └── db.py          # Simple database utilities
├── notebooks/
│   ├── AgentX_InitalPOC.ipynb    # Main demo notebook
│   └── dataprepartion.ipynb      # Data preprocessing
├── requirements.txt
└── README.md
```

## Usage

### Basic Usage

1. Start with the demo notebook:
```python
from src.types import GraphState
from src.graph import create_cultural_graph
from langgraph.checkpoint.memory import MemorySaver

# Create memory checkpointer
memory = MemorySaver()

# Initialize the graph
workflow = create_cultural_graph()
debator = workflow.with_config(run_name="Cultural Debate")

# Prepare input state
state = {
    "question_meta": {
        "original": "What are the gender roles in different cultures?"
    },
    "user_profile": {
        "id": "user_001",
        "demographics": {
            "age": 30,
            "location": "UK",
            "gender": "female"
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

# Get culturally-informed response
result = debator.invoke(state)
```

### Workflow Steps

1. **Cultural Sensitivity Check**: Determines if the input question requires cultural consideration
2. **Topic Extraction**: Identifies specific culturally sensitive components
3. **Culture Routing**: Selects relevant cultural perspectives to consult
4. **Expert Consultation**: Gathers responses from cultural experts
5. **Response Aggregation**: Combines expert insights into a comprehensive answer
6. **Final Composition**: Generates the final, culturally-informed response

## Extending the System

### Adding New Cultural Experts

1. Modify the cultures list in `src/graph.py`:
```python
cultures = ["US", "China", "India", "NewCulture"]
```

### Customizing the Pipeline

The graph structure can be modified in `src/graph.py` by:
- Adding new nodes for additional processing steps
- Modifying edge conditions for different routing logic
- Implementing new state transitions

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

[Your chosen license]

## Acknowledgments

- LangGraph for the workflow management framework
- LangChain for LLM integration
- Groq for LLM API access