#!/bin/bash
echo "=== 100-Cycle Run Progress ==="
echo "Time: $(date)"
echo ""

# Check current test number
if docker exec cultural-agent-optimized test -f /app/run_100_output.log; then
    CURRENT_TEST=$(docker exec cultural-agent-optimized grep -o "Test [0-9]*/100" /app/run_100_output.log | tail -1)
    echo "Current: $CURRENT_TEST"
    
    # Check for progress updates
    PROGRESS=$(docker exec cultural-agent-optimized grep ">>> PROGRESS:" /app/run_100_output.log | tail -1)
    if [ ! -z "$PROGRESS" ]; then
        echo "$PROGRESS"
    fi
    
    # Check log size
    SIZE=$(docker exec cultural-agent-optimized ls -lh /app/run_100_output.log | awk '{print $5}')
    echo "Log size: $SIZE"
    
    # Estimate time remaining based on 58s per test
    if [ ! -z "$CURRENT_TEST" ]; then
        CURRENT_NUM=$(echo $CURRENT_TEST | grep -o "[0-9]*" | head -1)
        REMAINING=$((100 - CURRENT_NUM))
        EST_MINUTES=$((REMAINING * 58 / 60))
        echo "Estimated time remaining: ~$EST_MINUTES minutes"
    fi
else
    echo "Run not started or log file not found"
fi

echo ""
echo "Check full progress with:"
echo "docker exec cultural-agent-optimized tail -f /app/run_100_output.log"