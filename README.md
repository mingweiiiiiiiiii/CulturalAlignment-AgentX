# Cultural Alignment Project

## Overview
The Cultural Alignment Project is designed to facilitate a culturally-aware multi-agent dialogue pipeline utilizing LangGraph. The primary objective is to analyze cultural sensitivities and generate responses that reflect diverse cultural perspectives.

## Installation

To set up the project, ensure you have the following packages installed:

### Using `pip` and `venv` (Standard)

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
    *(Use `python3` if `python` doesn't point to Python 3)*

2.  **Activate the environment:**
    *   macOS/Linux: `source venv/bin/activate`
    *   Windows (cmd): `.\venv\Scripts\activate`
    *   Windows (PowerShell): `.\venv\Scripts\Activate.ps1`

3.  **Install dependencies:**
    ```bash
    pip install pymilvus torch transformers matplotlib pyyaml
    ```
    *(Alternatively, if a `requirements.txt` file exists: `pip install -r requirements.txt`)*

### Using `uv` (Faster Alternative)

1.  **Install `uv`** (if you haven't already): Follow the official uv installation guide.

2.  **Create and activate the virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate  # macOS/Linux
    # .\.venv\Scripts\activate # Windows (cmd)
    # .\.venv\Scripts\Activate.ps1 # Windows (PowerShell)
    ```

3.  **Install dependencies using `uv`:**
    ```bash
    uv pip install -r requirements.txt
    ```

 ## Running the Project
 To initiate the dialogue pipeline, first ensure your `uv` virtual environment is active (`source .venv/bin/activate` or similar). Then, execute the following command in your terminal:

 ```bash
 uv run python main.py
 ```

## Project Structure
The project is organized as follows:

- **`main.py`**: The main entry point for the application, responsible for managing user profiles and executing the cultural analysis.

### Folders

- **`corpora`**: During development phase, the following files should be included in the folder

    - `persona_data_list.json`: Samples of persona data
    - `wvs_questions.json`: World Value Survey questions

- **`llmagentsetting`**: Include the LLM clients and configuration

- **`mylanggraph`**: Graph of Cultural Alignment System defined in LangGraph, and data structure's schema

- **`node`**: Detail implementation of the nodes in the pipeline

- **`utility`**: Utility function to process input data.


## Contributing
We welcome contributions! Feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. For more details, please refer to the LICENSE file.
