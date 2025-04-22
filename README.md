# Cultural Alignment Project

## Description
This project orchestrates a culturally-aware multi-agent dialogue pipeline using LangGraph. It aims to analyze cultural sensitivities and generate responses based on various cultural perspectives.

## Installation
To run this project, you need to install the following packages:

```bash
pip install pymilvus torch transformers matplotlib pyyaml
```

## Usage
To run the project, execute the following command:

```bash
python main.py
```

This will initiate the dialogue pipeline, sampling user profiles and questions, embedding parameters, and evaluating responses from cultural experts.

## File Structure
- `main.py`: Entry point for the application, managing user profiles and running the cultural analysis.
- `nodes.py`: Contains the definitions for cultural experts and their response generation logic.
- `db.py`: Handles interactions with the Milvus database for storing and retrieving cultural embeddings.
- `evaluation.py`: Evaluates the performance of cultural responses using various loss functions and metrics.
- `inputData.py`: Manages the sampling of user profiles and questions.
- `graph.py`: Defines the cultural graph for analyzing cultural sensitivities.
- `WorldValueSurveyProcess.py`: Contains functions related to embedding user profiles based on world values.
- `persona.yml`: YAML file containing persona templates for user sampling.
- `wvs_questions.json`: JSON file containing World Value Survey questions.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
