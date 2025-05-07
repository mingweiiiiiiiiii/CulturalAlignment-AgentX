# Cultural Alignment Project

## Overview
The Cultural Alignment Project is designed to facilitate a culturally-aware multi-agent dialogue pipeline utilizing LangGraph. The primary objective is to analyze cultural sensitivities and generate responses that reflect diverse cultural perspectives.

## Purpose and Significance

In an increasingly interconnected world, effective cross-cultural communication is paramount. However, digital interactions, especially those mediated by AI, often lack the nuanced understanding of diverse cultural contexts. This can lead to misinterpretations, alienation, and ineffective communication.

The Cultural Alignment Project aims to address this gap by:

-   **Enhancing AI's Cultural Intelligence:** Developing AI systems that can perceive, understand, and adapt to various cultural norms and sensitivities.
-   **Fostering Inclusive Dialogue:** Enabling AI-driven conversations that are respectful and considerate of diverse cultural backgrounds.
-   **Improving User Experience:** Providing more natural and effective interactions for users from different cultural upbringings.
-   **Advancing Research:** Contributing to the field of culturally-aware AI by exploring novel methods for integrating cultural knowledge into language models and agentic systems.
-   **Potential Applications:** This research can pave the way for applications in global customer service, international diplomacy, multicultural education, and content generation for diverse audiences.

## Methodology

The project employs a multi-agent dialogue pipeline built with LangGraph. This framework allows for a structured yet flexible interaction between specialized agents, each contributing to the cultural alignment process.

### Cultural Knowledge Integration

-   **World Value Survey (WVS):** Utilizes [WVS questions](https://www.worldvaluessurvey.org/wvs.jsp) to model and understand different cultural dimensions and values. This data helps in grounding the AI's understanding of cultural perspectives.
-   **Persona Data:** Incorporates diverse persona profiles in [SynthLabAI's dataset](https://huggingface.co/datasets/SynthLabsAI/PERSONA) representing individuals with varied cultural backgrounds and characteristics. These personas help simulate and test the system's ability to adapt to different cultural contexts.

### Key Technical Components & Process
1.  **Input & Cultural Contextualization:** User input is received. The system analyzes this input for cultural cues, potentially leveraging embeddings of cultural data (like WVS insights) and persona information.
2.  **Embedding Models:** The project uses embedding models (e.g., `mxbai-embed-large` served via Ollama) to convert textual data into dense vector representations. This enables semantic similarity searches and nuanced understanding of cultural nuances. The `pymilvus` dependency suggests these embeddings may be stored and queried efficiently using a vector database like Milvus.

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

### Prerequisite

- **ollama**: Currently this project use ollama to host embedding model, please make sure ollama is installed correctly, and pull `mxbai-embed-large` to be served.

- **`GEMINI API KEY`**: Gemini API key can be generated in [aistudio](https://aistudio.google.com/) after signing up for the service. The API key should be placed in .env (Please refer to .env.example, you can copy the file and put in API key accordingly and rename the file to .env)

- **`GROQ API KEY`**: Groq API key can be generated in [groq](https://groq.com/) after signing up for the service.(Please refer to .env.example, you can copy the file and put in API key accordingly and rename the file to .env

 To initiate the dialogue pipeline, first ensure your `uv` virtual environment is active (`source .venv/bin/activate` or similar). Then, execute the following command in your terminal:

 ```bash
 uv run python main.py
 ```

## Project Structure
The project is organized as follows:

- **`main.py`**: The main entry point for the application, responsible for managing user profiles and executing the cultural analysis.

- **`.env.example`**: This is an example file of how `.env` should look like, to run this project you need to obtain corresponding value separately and create a `.env` file following the format of `.env.example`

- **`requirements.txt`**: python libraries for the project

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
