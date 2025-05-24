#!/bin/bash
# Monitor GPU usage while running inference

echo "=== Testing Real GPU Usage ==="
echo "Starting background GPU monitor..."

# Start GPU monitoring in background
nvidia-smi dmon -s pucm -i 0 -d 1 > gpu_usage.log &
MONITOR_PID=$!

echo "Running inference test..."

# Run inference in container
docker exec cultural-agent-container python -c "
import ollama
import config
import time

client = ollama.Client(host=config.OLLAMA_HOST)

print('Starting inference...')
start = time.time()

response = client.generate(
    model='phi4',
    prompt='Write a detailed essay about the cultural significance of family values across different societies. Include specific examples from at least 5 different cultures.',
    options={'num_predict': 500}
)

elapsed = time.time() - start
print(f'Inference completed in {elapsed:.2f}s')
"

# Stop GPU monitoring
kill $MONITOR_PID 2>/dev/null

# Show results
echo -e "\n=== GPU Usage During Inference ==="
tail -20 gpu_usage.log | awk '{print "GPU Util: "$2"%, Power: "$3"W, Memory: "$4"%, SM Clock: "$5"MHz"}'

# Clean up
rm -f gpu_usage.log

echo -e "\n=== Current GPU Status ==="
nvidia-smi --query-gpu=gpu_name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv