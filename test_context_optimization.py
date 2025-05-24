#!/usr/bin/env python3
"""
Test phi4 performance with different context window sizes
"""
import time
import ollama
import config

def test_context_window_performance():
    print("=== Testing phi4 with Different Context Windows ===\n")
    
    client = ollama.Client(host=config.OLLAMA_HOST)
    
    # Test prompt
    prompt = """You are a cultural expert. Please provide a brief analysis of family values 
    in Chinese culture. Focus on the key aspects and be concise."""
    
    # Different context window sizes to test
    context_sizes = [2048, 4096, 8192, 16384]
    
    results = []
    
    for ctx_size in context_sizes:
        print(f"\nTesting with context window: {ctx_size}")
        
        # Time the inference
        start = time.time()
        
        try:
            response = client.generate(
                model="phi4",
                prompt=prompt,
                options={
                    "num_ctx": ctx_size,  # Set context window size
                    "num_predict": 150,   # Limit output length
                    "temperature": 0.7,
                    "num_thread": 8,      # Use multiple threads
                    "num_gpu": 999,       # Use all available GPU layers
                }
            )
            
            inference_time = time.time() - start
            
            # Extract response
            if hasattr(response, 'response'):
                text = response.response
            else:
                text = str(response)
            
            word_count = len(text.split())
            
            results.append({
                'context_size': ctx_size,
                'inference_time': inference_time,
                'words': word_count,
                'words_per_sec': word_count / inference_time
            })
            
            print(f"  Time: {inference_time:.2f}s")
            print(f"  Words: {word_count}")
            print(f"  Speed: {word_count/inference_time:.1f} words/sec")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'context_size': ctx_size,
                'inference_time': 0,
                'words': 0,
                'words_per_sec': 0
            })
    
    # Summary
    print("\n=== Performance Summary ===")
    print(f"{'Context':<10} {'Time (s)':<10} {'Words':<10} {'Words/sec':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['context_size']:<10} {r['inference_time']:<10.2f} {r['words']:<10} {r['words_per_sec']:<10.1f}")
    
    # Find optimal
    if results:
        best = max(results, key=lambda x: x['words_per_sec'])
        print(f"\n✅ Best performance: {best['context_size']} context window")
        print(f"   Speed: {best['words_per_sec']:.1f} words/sec")
        
        # Test with optimal settings for cultural agent use case
        print("\n=== Testing Optimized Settings for Cultural Agent ===")
        
        cultural_prompt = """From a Chinese cultural perspective, explain the concept of filial piety 
        and its importance in family relationships. Provide specific examples of how this manifests 
        in modern Chinese society."""
        
        start = time.time()
        response = client.generate(
            model="phi4",
            prompt=cultural_prompt,
            options={
                "num_ctx": best['context_size'],
                "num_predict": 200,
                "temperature": 0.7,
                "num_thread": 8,
                "num_gpu": 999,
                "top_k": 40,
                "top_p": 0.9,
            }
        )
        
        opt_time = time.time() - start
        
        if hasattr(response, 'response'):
            text = response.response
        else:
            text = str(response)
            
        words = len(text.split())
        
        print(f"\nOptimized cultural response:")
        print(f"  Time: {opt_time:.2f}s")
        print(f"  Words: {words}")
        print(f"  Speed: {words/opt_time:.1f} words/sec")
        print(f"\n  Response preview: {text[:200]}...")

if __name__ == "__main__":
    test_context_window_performance()