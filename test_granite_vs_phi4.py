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
print("GRANITE3.3 vs PHI4 PERFORMANCE COMPARISON")
print("=" * 80)
print("Testing cultural alignment system with different models")
print("Optimized settings: 8192 context, 8 threads, full GPU")
print("=" * 80)

# Test different models
models_to_test = [
    ("granite3.3:latest", "Granite 3.3 (4.9GB)"),
    ("phi4:latest", "Phi4 Q4_K_M (9.1GB)"),
    ("phi3:mini", "Phi3 Mini (2.2GB)")
]

results = []

for model_name, description in models_to_test:
    print(f"\n📊 Testing {description}...")
    
    try:
        # Warm up the model
        print("   Warming up...")
        client.generate(
            model=model_name,
            prompt="Hello",
            options={"num_predict": 1}
        )
        
        # Test with full prompt - 3 runs for accuracy
        times = []
        word_counts = []
        responses = []
        
        for run in range(3):
            print(f"   Run {run + 1}/3...")
            start_time = time.time()
            
            response = client.generate(
                model=model_name,
                prompt=test_prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": 150,
                    "num_ctx": 8192,
                    "num_thread": 8,
                    "num_gpu": 999,
                }
            )
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            # Extract response
            if hasattr(response, 'response'):
                response_text = response.response
            elif isinstance(response, dict) and 'response' in response:
                response_text = response['response']
            else:
                response_text = str(response)
            
            word_count = len(response_text.split())
            word_counts.append(word_count)
            responses.append(response_text)
        
        # Calculate averages
        avg_time = np.mean(times)
        avg_words = np.mean(word_counts)
        avg_wps = avg_words / avg_time if avg_time > 0 else 0
        
        results.append({
            "model": model_name,
            "description": description,
            "avg_time": avg_time,
            "avg_words": avg_words,
            "avg_wps": avg_wps,
            "best_response": responses[0],
            "success": True
        })
        
        print(f"   ✅ Average: {avg_time:.2f}s, {avg_words:.0f} words, {avg_wps:.1f} words/sec")
        print(f"   Response preview: {responses[0][:100]}...")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append({
            "model": model_name,
            "description": description,
            "success": False,
            "error": str(e)
        })

# Summary comparison
print(f"\n" + "=" * 80)
print("PERFORMANCE COMPARISON SUMMARY")
print("=" * 80)
print(f"{'Model':<25} {'Size':<10} {'Time (s)':<10} {'Words':<8} {'Words/sec':<12} {'Quality'}")
print("-" * 80)

successful_results = [r for r in results if r.get('success', False)]

for result in successful_results:
    model_size = "4.9GB" if "granite" in result['model'] else "9.1GB" if "phi4" in result['model'] else "2.2GB"
    quality = "✅ Good" if result['avg_wps'] > 5 else "⚠️ Slow"
    
    print(f"{result['description']:<25} {model_size:<10} {result['avg_time']:<10.2f} {result['avg_words']:<8.0f} {result['avg_wps']:<12.1f} {quality}")

# Find the best performer
if successful_results:
    # Best by speed
    fastest = min(successful_results, key=lambda x: x['avg_time'])
    fastest_wps = max(successful_results, key=lambda x: x['avg_wps'])
    
    print(f"\n🏆 Performance Winners:")
    print(f"   Fastest overall: {fastest['description']} ({fastest['avg_time']:.2f}s)")
    print(f"   Highest words/sec: {fastest_wps['description']} ({fastest_wps['avg_wps']:.1f} words/sec)")

# Quality comparison
print(f"\n📝 Response Quality Samples:")
for result in successful_results[:2]:  # Show first 2
    print(f"\n{result['description']}:")
    print(f"   {result['best_response'][:200]}...")

# Memory efficiency
print(f"\n💾 Model Size Comparison:")
print(f"   Granite3.3: 4.9GB (46% smaller than Phi4)")
print(f"   Phi4: 9.1GB (reference)")
print(f"   Phi3 Mini: 2.2GB (76% smaller than Phi4)")

print(f"\n🎯 Recommendations:")
if successful_results:
    granite_result = next((r for r in successful_results if "granite" in r['model']), None)
    phi4_result = next((r for r in successful_results if "phi4" in r['model']), None)
    
    if granite_result and phi4_result:
        speed_improvement = (phi4_result['avg_time'] - granite_result['avg_time']) / phi4_result['avg_time'] * 100
        size_improvement = (9.1 - 4.9) / 9.1 * 100
        
        if granite_result['avg_wps'] >= phi4_result['avg_wps']:
            print(f"   ✅ Switch to Granite3.3: {speed_improvement:.0f}% faster, {size_improvement:.0f}% smaller")
        else:
            print(f"   ⚖️ Trade-off: Granite3.3 is {size_improvement:.0f}% smaller but may be slower")
    else:
        print(f"   Use the fastest model that fits your GPU memory constraints")

print("=" * 80)