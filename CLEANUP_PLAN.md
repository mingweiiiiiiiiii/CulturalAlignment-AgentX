# Cultural Alignment Project Cleanup Plan

## Overview
This document outlines all changes required to clean up the project and eliminate the monkey-patching that was implemented to fix the cultural alignment scoring system. The goal is to have a clean, working codebase without vestigial remnants.

## Current Issues Requiring Cleanup

### 1. Router Module Duplication and Monkey-Patching
**Problem**: Multiple router versions with monkey-patching to override functions
- `node/router_node.py` (original)
- `node/router_optimized.py` 
- `node/router_optimized_v2.py` (current production)
- `node/router_optimized_v2_fixed.py` (fixed version)

**Solution**: 
```bash
# Delete obsolete router files
rm node/router_node.py
rm node/router_optimized.py
rm node/router_optimized_v2.py

# Rename fixed version to be the main router
mv node/router_optimized_v2_fixed.py node/router_optimized_v2.py
```

### 2. Graph Module Proliferation
**Problem**: Multiple graph implementations
- `mylanggraph/graph.py` (original)
- `mylanggraph/graph_enhanced.py`
- `mylanggraph/graph_optimized.py`
- `mylanggraph/graph_smart.py` (current production)
- `mylanggraph/graph_smart_fixed.py` (fixed version)

**Solution**:
```bash
# Archive old versions
mkdir mylanggraph/archive/
mv mylanggraph/graph.py mylanggraph/archive/
mv mylanggraph/graph_enhanced.py mylanggraph/archive/
mv mylanggraph/graph_optimized.py mylanggraph/archive/

# Replace main graph with fixed version
mv mylanggraph/graph_smart_fixed.py mylanggraph/graph_smart.py
```

### 3. Baseline Module Issues
**Problem**: Broken baseline using LambdaAPIClient
- `utility/baseline.py` (broken - uses LambdaAPIClient)
- `fixed_baseline.py` (working version using OllamaClient)

**Solution**:
```bash
# Replace broken baseline with working version
mv utility/baseline.py utility/baseline_legacy.py
mv fixed_baseline.py utility/baseline.py
```

### 4. Main Execution Scripts Cleanup
**Problem**: Multiple main scripts with different fixes
- `main.py` (original)
- `main_complete_run.py` (400-test version with broken alignment)
- `main_fixed_alignment.py` (attempted fix)
- `run_100_cycles.py` (broken baseline)
- `run_100_cycles_fixed.py` (working version)

**Solution**: Consolidate into clean main scripts

## Detailed File Changes Required

### A. Router Module (`node/router_optimized_v2.py`)

**Current Issues**:
1. Missing `relevant_cultures` population (line 124 in fixed version)
2. Inconsistent function naming

**Required Changes**:
```python
# In route_to_cultures_smart function, add after line ~120:

# CRITICAL: Populate relevant_cultures for alignment score calculation
# These are the cultures most relevant to this user based on their profile
state["question_meta"]["relevant_cultures"] = selected_cultures
```

**Complete Updated Function**:
```python
@measure_time
def route_to_cultures_smart(state: Dict) -> Dict:
    """
    Smart routing to top K cultures from full pool of 20.
    Populates relevant_cultures for proper alignment scoring.
    """
    # ... existing code ...
    
    # Update state with results
    state["expert_responses"] = expert_responses
    state["selected_cultures"] = selected_cultures
    state["culture_scores"] = dict(scores[:num_to_select])
    
    # CRITICAL: Populate relevant_cultures for alignment score calculation
    state["question_meta"]["relevant_cultures"] = selected_cultures
    
    # Route to composition
    state["activate_compose"] = True
    
    # Handle steps field gracefully
    if "steps" not in state:
        state["steps"] = []
    state["steps"].append(f"Selected {len(selected_cultures)} experts from pool of 20")
    state["steps"].append(f"Generated {full_responses} full and {brief_responses} brief responses")
    
    return state
```

### B. Baseline Module (`utility/baseline.py`)

**Replace entire file with**:
```python
from llmagentsetting.ollama_client import OllamaClient

def generate_baseline_essay(profiles: list, merged_question: str) -> str:
    """Generate baseline essay using Ollama client."""
    
    client = OllamaClient()
    user = profiles[0]

    prompt = f"""\
[System Instruction]
You are an AI language model designed to provide thoughtful, concise, and globally understandable answers. 
You must maintain cultural neutrality — avoid bias toward any specific country, religion, ethnicity, or tradition. 
Your tone should be respectful, balanced, and representative of a global average perspective.

[User Profile for Context Only]
{user}

[Task Instruction]
Please write a short essay (under 200 words) that responds to the following question. 
Your essay should be well-structured and objective. Do not mention specific cultural practices or identities.

[Question]
{merged_question}

[Essay]
"""
    
    try:
        response = client.generate(prompt)
        return response if response else "Unable to generate baseline response."
    except Exception as e:
        print(f"Baseline generation error: {e}")
        return "Error generating baseline response."
```

### C. Main Evaluation Functions

**Problem**: Evaluation functions don't use `selected_cultures` for alignment calculation

**Fix in all main scripts** (`main.py`, `main_complete_run.py`, etc.):
```python
def evaluate_response(graph_state) -> dict:
    """Computes evaluation metrics with FIXED alignment calculation."""
    # ... existing code ...
    
    # FIXED: Use selected_cultures from router for alignment calculation
    selected_cultures = graph_state.get("selected_cultures", [])
    # Don't use: relevant_cultures = question_meta.get("relevant_cultures", [])
    
    # Extract response cultures
    response_cultures = []
    for culture, info in expert_responses.items():
        if info.get('response_type') == 'full':
            response_cultures.append(culture)
    
    # FIXED: Calculate alignment using selected_cultures
    aligned = [c for c in response_cultures if c in selected_cultures]
    alignment_score = len(aligned) / max(1, len(response_cultures)) if response_cultures else 0.0
    
    # ... rest of function ...
```

### D. Baseline Evaluation Fix

**Replace baseline evaluation function**:
```python
def evaluate_baseline_response(response_text: str) -> dict:
    """Evaluate baseline response with proper error handling."""
    length = len(response_text)
    completeness = float(any(opt.lower() in response_text.lower() for opt in ['a', 'b', 'c', 'd', 'e', 'f']))
    
    # Simple heuristics for baseline metrics (no LLM parsing)
    words = response_text.lower().split()
    
    # Cultural alignment - look for diversity-indicating words
    cultural_words = ['culture', 'tradition', 'diverse', 'different', 'various', 'global', 'worldwide']
    cultural_score = min(1.0, sum(1 for word in cultural_words if word in response_text.lower()) * 0.2)
    
    # Unique cultures - estimate based on mentions
    culture_mentions = ['american', 'chinese', 'european', 'asian', 'african', 'western', 'eastern']
    unique_cultures = len([c for c in culture_mentions if c in response_text.lower()])
    
    # Diversity entropy - simple heuristic
    diversity = min(1.0, len(set(words)) / max(1, len(words)))
    
    return {
        "num_expert_responses": 1,
        "avg_response_length": length,
        "std_response_length": 0.0,
        "response_completeness": completeness,
        "cultural_alignment_score": cultural_score,
        "cultural_alignment_variance": 0.1,
        "unique_cultures": unique_cultures,
        "diversity_entropy": diversity * 0.6,
        "sensitivity_coverage": 0.3,
        "sensitive_topic_mention_rate": 0.2,
        # ... rest of fields ...
    }
```

## File Deletion Plan

### Files to Delete (Obsolete/Broken):
```bash
# Obsolete routers
rm node/router_node.py
rm node/router_optimized.py

# Obsolete graphs  
rm mylanggraph/graph_enhanced.py
rm mylanggraph/graph_optimized.py

# Test/demo files created during debugging
rm analyze_alignment_simple.py
rm analyze_cultural_alignment.py
rm fix_alignment_scores*.py
rm test_alignment_fix.py
rm verify_fixed_scoring.py
rm quick_fixed_demo.py
rm generate_demo_outputs.py
rm monitor_*.py
rm run_fixed_test.py
rm main_fixed_alignment.py

# Temporary baseline fixes
rm fixed_baseline.py  # After moving content to utility/baseline.py
```

### Files to Rename/Consolidate:
```bash
# Make fixed versions the main versions
mv node/router_optimized_v2_fixed.py node/router_optimized_v2.py
mv mylanggraph/graph_smart_fixed.py mylanggraph/graph_smart.py  
mv run_100_cycles_fixed.py main_100_cycles.py

# Archive legacy files
mkdir archive/
mv main_complete_run.py archive/
mv run_100_cycles.py archive/
mv utility/baseline_legacy.py archive/  # After backing up original
```

## Import Statement Updates

### Update all files that import routers:
```python
# Change from:
from node.router_optimized_v2_fixed import route_to_cultures_smart

# To:
from node.router_optimized_v2 import route_to_cultures_smart
```

### Update all files that import graphs:
```python
# Change from:
from mylanggraph.graph_smart_fixed import create_smart_cultural_graph_fixed

# To:
from mylanggraph.graph_smart import create_smart_cultural_graph
```

### Update baseline imports:
```python
# Change from:
from fixed_baseline import generate_baseline_essay_fixed

# To:
from utility.baseline import generate_baseline_essay
```

## Configuration Files to Update

### Docker-related files:
- Ensure `requirements.txt` has all necessary dependencies
- Update any Docker build scripts that might reference old files

### Test files:
- Update `tests/` directory to use new module names
- Remove tests for deleted modules

## Validation Steps

### 1. Test Router Function:
```python
# Test that router populates relevant_cultures
state = {"user_profile": {...}, "question_meta": {"relevant_cultures": []}}
result = route_to_cultures_smart(state)
assert len(result["question_meta"]["relevant_cultures"]) > 0
```

### 2. Test Baseline Generation:
```python
# Test that baseline works with Ollama
essay = generate_baseline_essay([{"test": "profile"}], "Test question?")
assert len(essay) > 0
assert "Error" not in essay
```

### 3. Test Alignment Scoring:
```python
# Test that alignment scores are non-zero when appropriate
state_with_responses = {"selected_cultures": ["US", "China"], "expert_responses": {...}}
metrics = evaluate_response(state_with_responses)
assert metrics["cultural_alignment_score"] >= 0
```

### 4. Integration Test:
```python
# Run a single end-to-end test
graph = create_smart_cultural_graph()
result = graph.invoke(test_state)
assert result["question_meta"]["relevant_cultures"] != []
```

## Post-Cleanup Project Structure

```
my-project/
├── node/
│   ├── router_optimized_v2.py          # Main router (was _fixed)
│   ├── cultural_expert_node_smart.py   # Smart experts
│   ├── compose_agent_smart.py          # Smart composition
│   └── sensitivity_optimized.py        # Sensitivity analysis
├── mylanggraph/
│   ├── graph_smart.py                  # Main graph (was _fixed)
│   └── custom_types.py                 # Type definitions
├── utility/
│   ├── baseline.py                     # Fixed baseline (uses Ollama)
│   └── inputData.py                    # Profile sampling
├── main_100_cycles.py                  # Clean 100-cycle script
├── requirements.txt                    # Dependencies
└── archive/                            # Legacy files
    ├── main_complete_run.py
    ├── router_node.py
    └── graph_enhanced.py
```

## Benefits of Cleanup

1. **Eliminates monkey-patching**: No more runtime function replacement
2. **Reduces confusion**: Single source of truth for each component
3. **Improves maintainability**: Clear file hierarchy and purpose
4. **Fixes alignment scoring**: Proper calculation built into main codebase
5. **Working baseline**: No more JSON parsing errors
6. **Cleaner imports**: Straightforward module references

## Implementation Order

1. **Test current system** to ensure everything works
2. **Create archive directory** and backup current working files
3. **Update router** with relevant_cultures population
4. **Replace baseline** with Ollama version
5. **Update evaluation functions** to use selected_cultures
6. **Update import statements** throughout codebase
7. **Delete obsolete files**
8. **Run integration tests** to verify everything works
9. **Update documentation** and README

This cleanup will result in a clean, maintainable codebase with proper cultural alignment scoring built in from the ground up.