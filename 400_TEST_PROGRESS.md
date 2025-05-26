# 400-Test Benchmark Progress

## Status
- **Started**: 2025-05-24 15:00:34
- **Total Tests**: 400
- **Estimated Duration**: ~6-7 hours (based on ~60 seconds per test)
- **Expected Completion**: Around 21:00-22:00 (9-10 PM)

## How to Monitor Progress

### Check current test number:
```bash
tail -20 /home/kyle/projects/my-project/main_400_tests_output.log | grep -E "Test [0-9]+/400"
```

### Check for progress milestones (every 50 tests):
```bash
grep "PROGRESS:" /home/kyle/projects/my-project/main_400_tests_output.log
```

### Check if still running:
```bash
ps aux | grep "python main_complete_run.py"
```

### View latest log entries:
```bash
tail -f /home/kyle/projects/my-project/main_400_tests_output.log
```

## Expected Outputs
When complete, the following files will be generated:
- `eval_results_[timestamp].csv` - Full evaluation metrics for 800 rows (400 model + 400 baseline)
- `paired_profiles_metrics_[timestamp].json` - 400 user profiles with performance data
- `correlation_analysis.zip` - Statistical analysis and visualizations
- `run.log` - Complete execution log

## Notes
- The run is executing in the background using `nohup`
- Each test includes:
  1. Cultural sensitivity analysis
  2. Expert selection (if sensitive)
  3. Response generation
  4. Baseline comparison
  5. Evaluation metrics
- Progress updates will be logged every 50 tests
- The cultural alignment system consults up to 5 experts from a pool of 20 cultures