#!/usr/bin/env python3
import ollama
import time
import config
import numpy as np

# Test prompt for cultural agent use case
test_prompt = """You are a cultural expert from China, deeply familiar with its historical, social, moral, and traditional nuances. 
Frame your answer considering the values, etiquette, common beliefs, communication styles, and societal norms typical of China. 
Include aspects like community vs individualism, indirect vs direct communication, formality levels, views on authority, spirituality, family roles, and social relationships. 
Be thoughtful, factual, respectful of diversity, and avoid generalizations or stereotypes. 
Keep the response under 150 words.

Question: 'What is your opinion on government policies regarding individual freedoms versus collective security?'"""

# Initialize Ollama client
client = ollama.Client(host=config.OLLAMA_HOST)

print("=" * 80)
print("PHI4 QUANTIZATION PERFORMANCE TEST")
print("=" * 80)

# Test different model configurations
models_to_test = [
    ("phi4:latest", "Default model"),
    ("phi4:14b-q4_K_M", "Q4_K_M quantized"),
    ("phi3:mini", "Phi3 mini for comparison")
]

results = []

for model_name, description in models_to_test:
    print(f"\n📊 Testing {description} ({model_name})...")
    
    try:
        # Warm up the model with a short prompt
        print("   Warming up...")
        client.generate(
            model=model_name,
            prompt="Hello",
            options={"num_predict": 1}
        )
        
        # Test with full prompt
        print("   Running test...")
        start_time = time.time()
        
        response = client.generate(
            model=model_name,
            prompt=test_prompt,
            options={
                "temperature": 0.7,
                "num_predict": 150,
                "num_ctx": 8192,
                "num_thread": 8,
                "num_gpu": 999,  # Use all GPU layers
            }
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract response
        if hasattr(response, 'response'):
            response_text = response.response
        elif isinstance(response, dict) and 'response' in response:
            response_text = response['response']
        else:
            response_text = str(response)
        
        word_count = len(response_text.split())
        words_per_second = word_count / elapsed_time if elapsed_time > 0 else 0
        
        results.append({
            "model": model_name,
            "description": description,
            "time": elapsed_time,
            "words": word_count,
            "wps": words_per_second,
            "response": response_text
        })
        
        print(f"   ✅ Completed in {elapsed_time:.2f}s")
        print(f"   📝 Generated {word_count} words ({words_per_second:.1f} words/sec)")
        print(f"   💬 Response preview: {response_text[:100]}...")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append({
            "model": model_name,
            "description": description,
            "time": 0,
            "words": 0,
            "wps": 0,
            "response": f"Error: {str(e)}"
        })

# Summary
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"{'Model':<30} {'Time (s)':<10} {'Words':<10} {'Words/sec':<10}")
print("-" * 70)

for result in results:
    print(f"{result['description']:<30} {result['time']:<10.2f} {result['words']:<10} {result['wps']:<10.1f}")

# Find the fastest model
if results:
    fastest = min(results, key=lambda x: x['time'] if x['time'] > 0 else float('inf'))
    print(f"\n🏆 Fastest: {fastest['description']} ({fastest['time']:.2f}s)")

# Memory usage check
print("\n💾 GPU Memory Usage:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        output = result.stdout.strip()
        used, total = map(int, output.split(', '))
        print(f"   {used} MB / {total} MB ({used/total*100:.1f}% used)")
except:
    print("   Unable to check GPU memory")