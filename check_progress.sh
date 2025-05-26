#!/bin/bash

echo "======================================"
echo "400-TEST BENCHMARK PROGRESS CHECK"
echo "======================================"

# Check if process is still running
if pgrep -f "python main_complete_run.py" > /dev/null; then
    echo "Status: RUNNING ✓"
else
    echo "Status: COMPLETED or STOPPED"
fi

# Check container logs for progress
echo ""
echo "Latest Progress:"
echo "----------------"
docker exec cultural-agent-optimized tail -n 20 /app/run.log | grep -E "(Test [0-9]+/400|PROGRESS:|Elapsed:|RUN COMPLETE)"

# Check output file
echo ""
echo "Output Log Size:"
ls -lh main_400_tests_output.log 2>/dev/null || echo "Output log not found yet"

# Estimate completion
echo ""
echo "Estimated Completion Time:"
echo "-------------------------"
echo "Based on ~60 seconds per test:"
echo "Total time: ~400 minutes (6.7 hours)"
echo "Started at: $(date)"

# Check for any generated files
echo ""
echo "Generated Files (if any):"
echo "-------------------------"
docker exec cultural-agent-optimized ls -la /app/*.csv /app/*.json /app/*.zip 2>/dev/null | grep "$(date +%Y-%m-%d)" || echo "No files generated yet today"