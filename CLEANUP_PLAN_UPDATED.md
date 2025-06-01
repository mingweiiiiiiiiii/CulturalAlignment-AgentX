# Project Cleanup Plan - Updated

This document outlines all changes required to eliminate monkey-patching and create a clean, maintainable codebase, **including changes made during the cultural alignment fix on 2025-05-26**.

## Current State Analysis (Updated)

The project has accumulated even more versions due to the cultural alignment scoring fixes:

### Router Implementations (5 versions now)
- `router_optimized_v2.py` - Original with broken alignment
- `router_optimized_v2_fixed.py` - Fixed to populate relevant_cultures (circular logic)
- `router_optimized_v2_fixed_proper.py` - Attempted proper fix (incomplete, unused)
- `router_optimized_v2_fixed_final.py` - **FINAL working version with protected user_relevant_cultures**
- Legacy router files in various test directories

### Cultural Alignment Infrastructure (NEW)
- `fix_cultural_alignment.py` - **NEW**: Culture mapping logic and alignment calculation
- Multiple location/ancestry/language mapping dictionaries
- `derive_relevant_cultures()` function
- `calculate_meaningful_alignment()` function

### Evaluation Scripts (12+ versions now)
- `run_100_cycles.py` - Original broken version
- `run_100_cycles_fixed.py` - Fixed baseline but circular alignment logic
- `run_100_cycles_fixed_proper.py` - Attempted proper fix (unused)
- `cultural_alignment_validator.py` - **FINAL working version with proper alignment**
- Various test and debug scripts from alignment fixing

### Additional Files Created Today
- `monitor_final_run.sh` - Progress monitoring script
- `monitor_100_progress.sh` - Earlier monitoring script
- Multiple correlation analysis outputs with different suffixes (_proper, _final)
- Test output files and logs

### Current Working Stack (Post-Fix)
**THESE ARE THE CANONICAL VERSIONS:**
- Router: `router_optimized_v2_fixed_final.py`
- Evaluation: `run_100_cycles_final.py` 
- Cultural Utils: `fix_cultural_alignment.py`
- Baseline: `fixed_baseline.py`
- Graph: `graph_smart.py`

## Updated Cleanup Strategy

### Phase 1: Consolidate Core Components (UPDATED)

#### 1.1 Router Consolidation
**Target**: Single `router_optimized.py` file

**Actions**:
- Rename `router_optimized_v2_fixed_final.py` → `router_optimized.py`
- **CRITICAL**: Ensure the final version has the protected `user_relevant_cultures` field
- Delete obsolete versions:
  - `router_optimized_v2.py`
  - `router_optimized_v2_fixed.py` 
  - `router_optimized_v2_fixed_proper.py`

**Code Changes**:
```python
# Update all imports throughout codebase:
# FROM: from node.router_optimized_v2_fixed_final import route_to_cultures_smart
# TO:   from node.router_optimized import route_to_cultures_smart
```

#### 1.2 Cultural Alignment Module (NEW)
**Target**: `utility/cultural_alignment.py`

**Actions**:
- Move content from `fix_cultural_alignment.py` to proper utility module
- Organize into logical sections:
  ```python
  # Cultural mappings
  LOCATION_TO_CULTURES = {...}
  ANCESTRY_TO_CULTURES = {...}
  LANGUAGE_TO_CULTURES = {...}
  
  # Core functions
  def derive_relevant_cultures(user_profile: dict) -> List[str]:
  def calculate_meaningful_alignment(expert_responses, selected_cultures, relevant_cultures) -> float:
  ```
- Delete `fix_cultural_alignment.py`

**Code Changes**:
```python
# Update imports:
# FROM: from fix_cultural_alignment import derive_relevant_cultures, calculate_meaningful_alignment
# TO:   from utility.cultural_alignment import derive_relevant_cultures, calculate_meaningful_alignment
```

#### 1.3 Graph Consolidation  
**Target**: Single `graph.py` file

**Actions**:
- Keep `graph_smart.py` as the main implementation
- Rename `graph_smart.py` → `graph.py`
- Update function name for consistency:
  - `create_smart_cultural_graph` → `create_cultural_graph`
- Delete obsolete versions

**Code Changes**:
```python
# Update imports:
# FROM: from mylanggraph.graph_smart import create_smart_cultural_graph
# TO:   from mylanggraph.graph import create_cultural_graph
```

#### 1.4 Baseline Consolidation
**Target**: Integrate into main utility module

**Actions**:
- Move `generate_baseline_essay_fixed` into `utility/baseline.py`
- Rename function: `generate_baseline_essay_fixed` → `generate_baseline_essay`
- Delete `fixed_baseline.py`

### Phase 2: Evaluation Script Cleanup (UPDATED)

#### 2.1 Main Evaluation Script
**Target**: Single `evaluation.py` file

**Actions**:
- Rename `run_100_cycles_final.py` → `evaluation.py`
- Add command-line arguments for test count
- Move evaluation functions to separate utility module
- Delete ALL obsolete versions:
  - `run_100_cycles.py`
  - `run_100_cycles_fixed.py`
  - `run_100_cycles_fixed_proper.py`

**Code Changes**:
```python
# Add argument parsing:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tests', type=int, default=100, help='Number of tests to run')
parser.add_argument('--output-dir', default='./results', help='Output directory')
args = parser.parse_args()
```

#### 2.2 Evaluation Utilities (NEW)
**Target**: `utility/evaluation.py`

**Actions**:
- Move `evaluate_response()` function from run scripts
- Move `evaluate_baseline_response_fixed()` → `evaluate_baseline_response()`
- Standardize evaluation interface
- Add proper type hints and documentation

### Phase 3: Remove Monkey-Patching (CRITICAL UPDATE)

#### 3.1 Current Monkey-Patching Locations
**Files containing monkey-patching**:
- `run_100_cycles_final.py` (lines 27-29)
- Any other evaluation scripts
- Potentially in graph construction

**Current Problem**:
```python
# Monkey-patching (BAD):
from node import router_optimized_v2
from node import router_optimized_v2_fixed_final
router_optimized_v2.route_to_cultures_smart = router_optimized_v2_fixed_final.route_to_cultures_smart
```

**Solution**:
```python
# Direct import (GOOD):
from node.router_optimized import route_to_cultures_smart
```

#### 3.2 Graph Construction Update
**Current**: Graph imports old router, gets monkey-patched
**Target**: Graph imports correct router directly

**Files to Update**:
- `mylanggraph/graph.py` (after consolidation)
- All evaluation scripts
- API endpoints (`api.py`, `api_optimized.py`, etc.)

### Phase 4: File Organization and Deletion (EXPANDED)

#### 4.1 Delete Obsolete Router Files
```bash
# Router versions (keep only the final working one temporarily)
rm node/router_optimized_v2.py
rm node/router_optimized_v2_fixed.py
rm node/router_optimized_v2_fixed_proper.py
# router_optimized_v2_fixed_final.py will be renamed to router_optimized.py
```

#### 4.2 Delete Obsolete Evaluation Files
```bash
# Evaluation versions (keep only the final working one temporarily)
rm run_100_cycles.py
rm run_100_cycles_fixed.py
rm run_100_cycles_fixed_proper.py
# run_100_cycles_final.py will be renamed to evaluation.py
```

#### 4.3 Delete Cultural Alignment Development Files
```bash
# Cultural alignment development files
rm fix_cultural_alignment.py  # After moving to utility/cultural_alignment.py
```

#### 4.4 Delete Monitoring and Debug Files
```bash
# Monitoring scripts (development only)
rm monitor_final_run.sh
rm monitor_100_progress.sh

# Debug and test files
rm debug_*.py
rm test_*.py (development versions, keep proper test suite)
```

#### 4.5 Archive Results from Development Iterations
```bash
# Archive old results with various suffixes
mkdir -p archive/development_results
mv eval_results_*proper*.csv archive/development_results/
mv eval_results_*final*.csv archive/development_results/
mv paired_profiles_*proper*.json archive/development_results/
mv paired_profiles_*final*.json archive/development_results/
mv correlation_analysis_*.zip archive/development_results/
mv model_vs_baseline_comparison_*.csv archive/development_results/
```

### Phase 5: Import Updates Throughout Codebase

#### 5.1 Files Requiring Import Updates
**Primary Files**:
- `api.py`
- `api_optimized.py` 
- `main.py`
- All remaining evaluation scripts
- Test files in `tests/`

**Import Changes**:
```python
# Router imports
# OLD: from node.router_optimized_v2_fixed_final import route_to_cultures_smart
# NEW: from node.router_optimized import route_to_cultures_smart

# Cultural alignment imports  
# OLD: from fix_cultural_alignment import derive_relevant_cultures, calculate_meaningful_alignment
# NEW: from utility.cultural_alignment import derive_relevant_cultures, calculate_meaningful_alignment

# Graph imports
# OLD: from mylanggraph.graph_smart import create_smart_cultural_graph
# NEW: from mylanggraph.graph import create_cultural_graph

# Baseline imports
# OLD: from fixed_baseline import generate_baseline_essay_fixed
# NEW: from utility.baseline import generate_baseline_essay

# Evaluation imports
# OLD: Functions scattered in run scripts
# NEW: from utility.evaluation import evaluate_response, evaluate_baseline_response
```

### Phase 6: Documentation Updates (CRITICAL)

#### 6.1 Document Cultural Alignment System
**Target**: `docs/cultural_alignment.md`

**Content**:
- How cultural relevance is derived from user profiles
- Location/ancestry/language mapping logic
- Alignment score calculation methodology
- Why the protected `user_relevant_cultures` field is necessary

#### 6.2 Update Architecture Documentation
**Target**: `docs/architecture.md`

**Content**:
- Final system architecture after cleanup
- Component interaction diagram
- Data flow through cultural alignment pipeline

#### 6.3 Update README.md
- Document proper usage of evaluation script
- Explain cultural alignment scoring
- Update Docker instructions

### Phase 7: Validation and Testing (EXPANDED)

#### 7.1 Cultural Alignment Tests
**Target**: `tests/test_cultural_alignment.py`

**Test Cases**:
```python
def test_derive_relevant_cultures():
    # Test various profile combinations
    # Test edge cases (missing data, etc.)
    
def test_calculate_meaningful_alignment():
    # Test different expert/culture combinations
    # Test edge cases (no experts, no relevant cultures)
    
def test_protected_user_cultures():
    # Ensure user_relevant_cultures isn't overwritten
    # Test that alignment calculation uses correct field
```

#### 7.2 Integration Tests
**Target**: `tests/test_integration_fixed.py`

**Test Cases**:
- Full workflow with cultural alignment
- Verify no monkey-patching occurs
- Check output file generation with proper alignment scores

### Phase 8: Performance and Optimization

#### 8.1 Culture Mapping Optimization
- Move large dictionaries to external files (JSON/YAML)
- Implement caching for culture lookups
- Optimize embedding calculations in router

#### 8.2 Evaluation Performance
- Parallelize baseline generation
- Cache common embeddings
- Optimize file I/O operations

## Implementation Priority (UPDATED)

### Priority 1 (CRITICAL - Must be done first)
1. **Cultural alignment module creation** (`utility/cultural_alignment.py`)
2. **Router consolidation** (eliminate monkey-patching)
3. **Update all imports** (fix broken references)
4. **Test the consolidated system** (ensure alignment scoring works)

### Priority 2 (High - Clean up development artifacts)
5. **Evaluation script consolidation**
6. **Delete obsolete files** (router versions, run scripts)
7. **Archive development results**
8. **Update main API endpoints**

### Priority 3 (Medium - Polish and documentation)
9. **Graph consolidation and renaming**
10. **Documentation updates**
11. **Test suite creation**
12. **Performance optimization**

## Risk Mitigation (UPDATED)

### Critical Risks Added
1. **Cultural alignment regression**: Ensure the protected `user_relevant_cultures` field logic is preserved
2. **Import chain breakage**: Many files depend on the current import structure
3. **Results continuity**: Ensure new system produces equivalent alignment scores

### Validation Steps (UPDATED)
1. **Run cultural alignment test**: Verify scores are not 0.0/1.0 binary
2. **Compare before/after**: Run sample evaluation before and after cleanup
3. **Check all imports**: Ensure no import errors after consolidation
4. **Verify Docker build**: Container must build and run correctly

## Success Criteria (UPDATED)

### Code Quality
- Zero monkey-patching ✓
- Single source of truth for each component ✓
- Clear import hierarchy ✓
- **Protected cultural alignment logic** ✓

### Functionality  
- **Cultural alignment scores show realistic variation** (not just 0.0/1.0) ✓
- All existing features work ✓
- Performance maintained or improved ✓
- Output files generated correctly ✓

### Maintainability
- **Cultural alignment system is modular and testable** ✓
- New features easy to add ✓
- Clear separation of concerns ✓
- Comprehensive documentation ✓

## Final Notes

The cultural alignment fixes represent the most critical functionality in the system. **The cleanup must preserve the logic that prevents `relevant_cultures` from being overwritten by sensitivity analysis**, as this was the core issue that caused meaningless binary alignment scores.

All cleanup phases should include testing to ensure cultural alignment continues to work correctly with realistic score distributions.