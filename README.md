# Cultural Alignment Project

## Overview
The Cultural Alignment Project is built to be a culturally-aware multi-agent dialogue pipeline built with LangGraph. The system analyzes cultural sensitivities and generates responses that reflect diverse cultural perspectives, demonstrating **69%+ improvement** in cultural alignment over baseline approaches.

### **Key Achievements**
- **Clean Architecture**: No monkey-patching required - fully modular design
- **Proven Performance**: 69%+ better cultural alignment than baseline systems
- **Comprehensive Validation**: 10+ cycle testing with detailed metrics and analysis
- **Docker-Ready**: Complete containerized environment with GPU support
- **Production-Grade**: Robust error handling, logging, and monitoring capabilities

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

#### **Clean Architecture Design**
The system employs a modular, clean architecture without monkey-patching:

1. **Cultural Sensitivity Analysis**: Automatically detects culturally sensitive topics using advanced LLM analysis
2. **Smart Expert Selection**: Dynamically selects 2-4 most relevant cultural experts from a pool of 20+ cultures
3. **Intelligent Response Generation**: Generates full responses for highly relevant cultures, brief responses for others
4. **Cultural Alignment Scoring**: Measures how well responses align with user's cultural context

![graph](img/graph.png)

#### **Technical Implementation**
- **Embedding Models**: Uses `mxbai-embed-large` via Ollama for semantic similarity and cultural relevance
- **LangGraph Pipeline**: Structured multi-agent workflow with conditional routing
- **Smart Cultural Pool**: 20+ diverse cultures with intelligent selection algorithms
- **Validation Framework**: Comprehensive testing with model vs baseline comparison

## Installation & Setup

### **Recommended: Docker Setup (GPU-Enabled)**

The easiest way to run the cultural alignment system is using our pre-configured Docker environment:

1. **Prerequisites:**
   - Docker and Docker Compose
   - NVIDIA GPU drivers (for optimal performance)

2. **Quick Start:**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd cultural-alignment-project

   # Create environment file
   cp .env.example .env

   # Start the containerized environment
   ./run_docker.sh
   ```

3. **Verify Setup:**
   ```bash
   # Check Ollama service
   docker-compose exec ollama-gpu curl http://localhost:11434/api/version

   # Test the cultural alignment system
   docker exec -it cultural-agent-container python main.py
   ```

### **Alternative: Local Python Setup**

For local development without Docker:

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # .\venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Ollama locally:**
   ```bash
   # Install Ollama CLI
   # Pull required models
   ollama pull mxbai-embed-large
   ollama pull granite3-dense:8b
   ```

## Environment Configuration

1. Copy the `.env.example` file to `.env`:

   ```pwsh
   Copy-Item .env.example .env
   ```

2. Edit the `.env` file and set the required variables:
   - `OLLAMA_HOST`: URL of the Ollama service (e.g., `http://ollama-gpu:11434`)
   - Optional API keys:
     - `GEMINI_API_KEY`
     - `GROQ_API_KEY`
     - `HF_API_KEY`
     - `LAMBDA_API_KEY`

3. Save the `.env` file. Your environment is now configured for local development.

## Running the Project

### **Interactive Cultural Dialogue**
Run the main interactive system:

```bash
# Docker environment
docker exec -it cultural-agent-container python main.py

# Local environment
python main.py
```

### **Validation & Testing**

#### **Comprehensive System Validation**
Run the full validation suite to test cultural alignment performance:

```bash
# Docker environment (recommended)
docker exec -it cultural-agent-container python cultural_alignment_validator.py

# Local environment
python cultural_alignment_validator.py
```

**Validation Outputs:**
- `eval_results_*.csv` - Detailed test results with metrics
- `paired_profiles_metrics_*.json` - User profiles and cultural alignment data
- `correlation_analysis_*.zip` - Statistical analysis and visualizations
- `model_vs_baseline_comparison_*.csv` - Performance comparison table
- `run_final.log` - Execution logs

#### **Quick Baseline Test**
Test the baseline alignment scoring improvements:

```bash
docker exec -it cultural-agent-container python test_baseline_alignment_fix.py
```

### **Expected Performance**
- **Cultural Alignment Score**: 0.25-0.50 (model) vs 0.10-0.20 (baseline)
- **Expert Responses**: 2-4 experts per sensitive question
- **Processing Time**: ~13-15 seconds per question (including expert consultation)
- **Improvement**: **69%+ better cultural alignment** than baseline

## System Architecture

### **Core Components**

#### **Cultural Alignment Validator** (`cultural_alignment_validator.py`)
- **Purpose**: Comprehensive validation and evaluation of cultural alignment performance
- **Features**: Model vs baseline comparison, statistical analysis, performance metrics
- **Outputs**: CSV reports, JSON data, correlation analysis, comparison tables

#### **Smart Cultural Graph** (`mylanggraph/graph_smart.py`)
- **Purpose**: Main workflow orchestration using LangGraph
- **Features**: Conditional routing, expert selection, response composition
- **Architecture**: Clean, modular design without monkey-patching

#### **Cultural Expert Nodes** (`node/cultural_expert_node_smart.py`)
- **Purpose**: Generate culturally-aware responses from different perspectives
- **Features**: Dynamic expert pool, relevance-based selection, response optimization

#### **Sensitivity Analysis** (`node/enhanced_sensitivity_node.py`)
- **Purpose**: Detect culturally sensitive topics automatically
- **Features**: Advanced LLM analysis, threshold-based routing, topic classification

### **Data Flow**
1. **Input Processing**: User question and profile analysis
2. **Sensitivity Detection**: Automatic cultural sensitivity scoring
3. **Expert Selection**: Smart selection of 2-4 most relevant cultural experts
4. **Response Generation**: Full responses for relevant cultures, brief for others
5. **Final Composition**: Culturally-aligned response synthesis
6. **Validation**: Comprehensive metrics and performance evaluation

## Project Structure

### **Core Files**
- **`main.py`**: Interactive cultural dialogue system entry point
- **`cultural_alignment_validator.py`**: Comprehensive validation and evaluation script
- **`requirements.txt`**: Python dependencies
- **`.env.example`**: Environment configuration template
- **`docker-compose.yml`**: Docker container orchestration
- **`run_docker.sh`**: Docker startup script

### **Key Directories**

#### **`mylanggraph/`** - LangGraph Workflow
- `graph_smart.py`: Main cultural alignment workflow (clean architecture)
- `custom_types.py`: Type definitions and data structures
- `types.py`: Additional type definitions

#### **`node/`** - Pipeline Components
- `cultural_expert_node_smart.py`: Smart cultural expert response generation
- `enhanced_sensitivity_node.py`: Advanced cultural sensitivity detection
- `router_optimized_v2.py`: Clean expert selection and routing
- `compose_agent_smart.py`: Final response composition

#### **`utility/`** - Support Functions
- `inputData.py`: Persona and question sampling from WVS data
- `baseline.py`: Baseline response generation for comparison
- `cultural_alignment.py`: Cultural alignment scoring and metrics

#### **`llmagentsetting/`** - LLM Configuration
- `ollama_client.py`: Ollama service integration
- `llm_clients.py`: Multiple LLM provider support

#### **`tests/`** - Testing Framework
- Comprehensive test suite with Docker integration
- Unit tests for all major components
- Integration tests for full workflow

### **Data Sources**
- **`corpora/wvs_questions.json`**: World Values Survey questions for cultural analysis
- **Persona Data**: SynthLabAI dataset for diverse cultural profiles

## Performance & Validation Results

### **Proven Performance Metrics**
Our comprehensive validation demonstrates significant improvements in cultural alignment:

| **Metric** | **Model Performance** | **Baseline Performance** | **Improvement** |
|------------|----------------------|--------------------------|-----------------|
| **Cultural Alignment Score** | 0.283 ± 0.172 | 0.167 ± 0.044 | **+69%** |
| **Expert Responses** | 2.4 ± 1.43 | 1.0 ± 0.0 | **+140%** |
| **Response Diversity** | 1.234 ± 0.732 | 0.437 ± 0.022 | **+182%** |
| **Processing Time** | 13.6s ± 3.1s | 5.9s ± 1.8s | +7.7s overhead |

### **Validation Features**
- ✅ **10+ Cycle Testing**: Comprehensive validation with configurable test cycles
- ✅ **Statistical Analysis**: Correlation analysis, distribution plots, performance metrics
- ✅ **Model vs Baseline**: Direct comparison showing clear improvements
- ✅ **Cultural Sensitivity Detection**: 80% accuracy in identifying sensitive topics
- ✅ **Expert Selection**: Smart selection of 2-4 most relevant cultural experts
- ✅ **Clean Architecture**: No monkey-patching, fully modular design

### **Output Examples**
The system generates comprehensive reports including:
- **CSV Reports**: Detailed metrics for each test cycle
- **JSON Data**: User profiles paired with cultural alignment scores
- **Visualizations**: Correlation matrices, distribution plots, performance charts
- **Comparison Tables**: Model vs baseline performance analysis

## Contributing
We welcome contributions! Feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. For more details, please refer to the LICENSE file.
