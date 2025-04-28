# Cultural Alignment Project

## Overview
The Cultural Alignment Project is designed to facilitate a culturally-aware multi-agent dialogue pipeline utilizing LangGraph. The primary objective is to analyze cultural sensitivities and generate responses that reflect diverse cultural perspectives.

## Installation
To set up the project, ensure you have the following packages installed:

```bash
pip install pymilvus torch transformers matplotlib pyyaml
```

## Running the Project
To initiate the dialogue pipeline, execute the following command in your terminal:

```bash
python main.py
```

This command will start the dialogue pipeline, sampling user profiles and questions, embedding parameters, and evaluating responses from cultural experts.

## Project Structure
The project is organized as follows:

- **`main.py`**: The main entry point for the application, responsible for managing user profiles and executing the cultural analysis.
- **`nodes.py`**: Contains definitions for cultural experts and the logic for generating their responses.

- **`inputData.py`**: Handles the sampling of user profiles and questions.
- **`graph.py`**: Defines the cultural graph used for analyzing cultural sensitivities.


## Contributing
We welcome contributions! Feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. For more details, please refer to the LICENSE file.
