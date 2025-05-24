#!/usr/bin/env python3
"""
Test GPU performance with Ollama
"""
import time
import subprocess
import ollama
import config

def monitor_gpu():
    """Get current GPU memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return 0

def test_gpu_performance():
    print("=== Testing Ollama GPU Performance ===\n")
    
    client = ollama.Client(host=config.OLLAMA_HOST)
    
    # Test 1: Simple inference timing
    print("1. Testing inference speed...")
    test_prompt = "What is the meaning of life? Give a detailed philosophical answer."
    
    # Get baseline GPU memory
    gpu_mem_before = monitor_gpu()
    print(f"   GPU memory before: {gpu_mem_before} MB")
    
    # Time the inference
    start = time.time()
    response = client.generate(
        model="phi4",
        prompt=test_prompt,
        options={"num_predict": 200}
    )
    inference_time = time.time() - start
    
    # Check GPU memory during inference
    gpu_mem_after = monitor_gpu()
    print(f"   GPU memory after: {gpu_mem_after} MB")
    print(f"   Memory increase: {gpu_mem_after - gpu_mem_before} MB")
    
    # Extract response
    if hasattr(response, 'response'):
        text = response.response
    else:
        text = str(response)
    
    word_count = len(text.split())
    print(f"\n   Inference time: {inference_time:.2f}s")
    print(f"   Generated {word_count} words")
    print(f"   Speed: {word_count/inference_time:.1f} words/second")
    
    # Test 2: Check model info
    print("\n2. Checking model configuration...")
    try:
        show_response = client.show(model="phi4")
        if hasattr(show_response, 'modelfile'):
            print(f"   Model info: {show_response.modelfile[:200]}...")
        else:
            print(f"   Model info: {str(show_response)[:200]}...")
    except Exception as e:
        print(f"   Could not get model info: {e}")
    
    # Test 3: Multiple parallel requests (simulate expert queries)
    print("\n3. Testing parallel inference...")
    prompts = [
        "From a Chinese cultural perspective, what are family obligations?",
        "From an American perspective, what is individual freedom?",
        "From an Indian perspective, what is the role of tradition?"
    ]
    
    import concurrent.futures
    
    def generate_response(prompt):
        start = time.time()
        resp = client.generate(model="phi4", prompt=prompt, options={"num_predict": 150})
        elapsed = time.time() - start
        return elapsed
    
    print("   Running 3 parallel inferences...")
    par_start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(generate_response, p) for p in prompts]
        times = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    par_total = time.time() - par_start
    
    print(f"   Individual times: {[f'{t:.2f}s' for t in times]}")
    print(f"   Total parallel time: {par_total:.2f}s")
    print(f"   Average per query: {sum(times)/len(times):.2f}s")
    
    # Check if we're CPU or GPU bound
    print("\n4. Performance Analysis:")
    if inference_time > 5:
        print("   ⚠️  Inference seems slow for GPU acceleration")
        print("   Expected GPU speed: 50-100+ words/second")
        print("   Your speed: {:.1f} words/second".format(word_count/inference_time))
        print("\n   Possible issues:")
        print("   - Model might be too large for GPU VRAM")
        print("   - Not all layers loaded to GPU")
        print("   - CPU fallback due to memory constraints")
    else:
        print("   ✅ GPU acceleration appears to be working")

if __name__ == "__main__":
    test_gpu_performance()