#!/usr/bin/env python3
"""Monitor progress of the 100-cycle run."""
import os
import time
from datetime import datetime

def monitor_100_cycle_run():
    print("Monitoring 100-Cycle Run Progress")
    print("="*60)
    
    # Check log file
    log_file = "run.log"
    output_log = "run_100_output.log"
    
    # Check if process is running
    if os.path.exists(log_file):
        # Get file size and modification time
        size = os.path.getsize(log_file) / 1024  # KB
        mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
        print(f"Log file: {size:.1f} KB, last updated: {mtime}")
        
        # Count completed tests
        with open(log_file, 'r') as f:
            content = f.read()
            
        test_count = content.count("Test ") - content.count("Test completed")
        progress_lines = [line for line in content.split('\n') if ">>> PROGRESS:" in line]
        
        if progress_lines:
            print(f"\nLatest progress: {progress_lines[-1].strip()}")
        else:
            print(f"\nTests started: ~{test_count}")
        
        # Check for errors
        error_count = content.count("ERROR")
        if error_count > 0:
            print(f"Errors encountered: {error_count}")
        
        # Show recent activity
        lines = content.split('\n')
        recent_tests = [line for line in lines[-20:] if "Test " in line and "/100" in line]
        if recent_tests:
            print(f"\nRecent tests:")
            for test in recent_tests[-3:]:
                print(f"  {test.strip()}")
    
    # Check output files
    print("\n\nOutput files status:")
    expected_files = [
        ("eval_results_*.csv", "Evaluation results"),
        ("paired_profiles_metrics_*.json", "Paired profiles"),
        ("correlation_analysis.zip", "Correlation analysis"),
        ("model_vs_baseline_comparison.csv", "Comparison table"),
        ("run.log", "Execution log")
    ]
    
    import glob
    for pattern, desc in expected_files:
        matches = glob.glob(pattern)
        if matches:
            latest = max(matches, key=os.path.getmtime)
            size = os.path.getsize(latest) / 1024
            print(f"  ✓ {desc}: {latest} ({size:.1f} KB)")
        else:
            print(f"  ✗ {desc}: Not yet generated")
    
    # Estimate completion
    if os.path.exists(output_log):
        print(f"\n\nChecking output log...")
        with open(output_log, 'r') as f:
            output_content = f.read()
        
        if "FULL 100-CYCLE RUN COMPLETE" in output_content:
            print("✅ Run completed successfully!")
        elif "All tests completed" in output_content:
            print("✅ Tests completed, generating final outputs...")
        else:
            # Try to estimate time remaining
            completed = output_content.count("Test ") - output_content.count("Test completed")
            if completed > 0 and completed < 100:
                print(f"Estimated progress: {completed}/100 tests")
                print(f"Estimated time remaining: {(100-completed) * 1.5} minutes")

if __name__ == "__main__":
    monitor_100_cycle_run()