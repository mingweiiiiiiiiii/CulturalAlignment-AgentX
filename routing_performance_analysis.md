# Cultural Routing Performance Analysis

## Why Cultural Routing Takes 60.4 Seconds

The "routing" time reported includes not just the routing algorithm, but also the **expert response generation** that happens within the routing workflow. Here's the breakdown:

### 1. **Actual Routing Algorithm: ~0.4 seconds**
   - User profile embedding: 0.021s
   - Topic embeddings: 0.058s  
   - Expert embeddings (6 experts): 0.248s
   - Scoring and selection: ~0.08s
   - **Total: ~0.407s**

### 2. **Expert Response Generation: ~60 seconds**
   - After routing selects 3 experts (China, USA, India)
   - Each expert calls the LLM to generate a culturally-specific response
   - Each LLM call takes ~7-20 seconds (depending on response length)
   - With 3 experts: 3 × ~20s = **~60 seconds**

## The Confusion

The timing is reported under "route_to_cultures" because:
1. The routing node selects which experts to consult
2. The compose node then queries those experts for responses
3. The time for expert responses gets attributed to the routing phase

## Performance Bottlenecks

### Current Flow:
```
1. Route to cultures (0.4s) → Selects 3 experts
2. Generate expert responses (60s) → 3 LLM calls @ ~20s each  
3. Compose final response (36s) → 1 LLM call to synthesize
```

### Why Each LLM Call Takes So Long:
- Using Ollama with phi4 model
- Each call generates 100-150 words
- Network latency between containers
- Model inference time on CPU (no GPU acceleration in current setup)

## Optimization Suggestions

### 1. **Parallel Expert Queries**
   - Currently: Sequential (20s + 20s + 20s = 60s)
   - Optimized: Parallel (max(20s, 20s, 20s) = 20s)
   - Potential savings: 40 seconds

### 2. **Reduce Expert Count**
   - Use 2 experts instead of 3
   - Potential savings: 20 seconds

### 3. **Cache Expert Responses**
   - For common questions/cultures
   - Potential savings: 60 seconds for cached queries

### 4. **Use Faster Models**
   - Switch to smaller/faster models for expert responses
   - Use GPU acceleration if available

### 5. **Pre-compute Expert Embeddings**
   - Already implemented (caching works)
   - Saves ~0.25s per request

## Code Location of Bottleneck

The actual expert response generation happens in:
- `node/compose_agent.py` lines 38-44
- Each call to `expert.generate_response(question)` takes ~20 seconds

This is why the "routing" appears to take so long - it's actually measuring the time to both route AND generate expert responses.