#!/usr/bin/env python3
"""
Monitor GPU usage during inference
"""
import time
import subprocess
import threading
import ollama
import config

class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.max_util = 0
        self.max_memory = 0
        self.samples = []
        
    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        self.thread.join()
        
    def _monitor(self):
        while self.monitoring:
            try:
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 3:
                    util = int(parts[0])
                    mem = int(parts[1])
                    temp = int(parts[2])
                    
                    self.samples.append({
                        'util': util,
                        'memory': mem,
                        'temp': temp
                    })
                    
                    if util > self.max_util:
                        self.max_util = util
                    if mem > self.max_memory:
                        self.max_memory = mem
                        
            except:
                pass
            
            time.sleep(0.1)  # Sample every 100ms

def test_gpu_utilization():
    print("=== Monitoring GPU During Inference ===\n")
    
    client = ollama.Client(host=config.OLLAMA_HOST)
    monitor = GPUMonitor()
    
    # Long prompt to ensure GPU usage
    prompt = """You are a cultural expert. Please provide a comprehensive analysis of family values 
    across different cultures, including specific examples from Chinese, American, Indian, Japanese, 
    and European cultures. Discuss how these values have evolved over time and their impact on 
    modern society. Please be detailed and thorough in your response."""
    
    print("Starting GPU monitoring...")
    monitor.start()
    
    print("Running inference...")
    start = time.time()
    
    response = client.generate(
        model="phi4",
        prompt=prompt,
        options={
            "num_predict": 300,
            "temperature": 0.7
        }
    )
    
    inference_time = time.time() - start
    monitor.stop()
    
    # Analyze results
    print(f"\nInference completed in {inference_time:.2f}s")
    print(f"\nGPU Statistics:")
    print(f"- Max GPU Utilization: {monitor.max_util}%")
    print(f"- Max Memory Used: {monitor.max_memory} MB")
    print(f"- Samples collected: {len(monitor.samples)}")
    
    if monitor.samples:
        avg_util = sum(s['util'] for s in monitor.samples) / len(monitor.samples)
        print(f"- Average GPU Utilization: {avg_util:.1f}%")
        
        # Show utilization graph
        print("\nGPU Utilization Timeline:")
        for i, sample in enumerate(monitor.samples[::5]):  # Every 5th sample
            bar = '█' * (sample['util'] // 5)
            print(f"{i*0.5:.1f}s: {bar} {sample['util']}%")
    
    # Get response length
    if hasattr(response, 'response'):
        text = response.response
    else:
        text = str(response)
    
    tokens = len(text.split())
    print(f"\nGenerated {tokens} words at {tokens/inference_time:.1f} words/sec")
    
    # Diagnosis
    print("\n=== Diagnosis ===")
    if monitor.max_util < 50:
        print("⚠️  Low GPU utilization detected!")
        print("Possible causes:")
        print("- CPU bottleneck (data transfer)")
        print("- Model not fully optimized for GPU")
        print("- Memory bandwidth limitations")
    elif tokens/inference_time < 30:
        print("⚠️  Slow inference despite GPU usage!")
        print("Possible causes:")
        print("- Model is very large (14.66B parameters)")
        print("- Limited GPU memory causing swapping")
        print("- Suboptimal batch size")

if __name__ == "__main__":
    test_gpu_utilization()