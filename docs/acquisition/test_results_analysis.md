# Acquisition Test Results Analysis

A utility script for analyzing acquisition system test results.

## Purpose

The `analyze_test_results.py` script parses and analyzes the output from acquisition system tests, providing summary statistics and identifying issues with camera synchronization, frame timing, and overall performance.

## Usage

```bash
python tests/acquisition/analyze_test_results.py [test_matrix_results.json]
```

If no file is specified, it looks for `test_matrix_results.json` in the current directory.

## Output

The script generates several reports:

1. **Overall Summary** - Total test runs and configurations
2. **Test Matrix** - Pass/fail matrix showing results by camera count and frame count
3. **Configuration Summary** - Average metrics grouped by configuration
4. **Failure Analysis** - Detailed breakdown of any failed tests

## Pass/Fail Criteria

Tests are evaluated against these thresholds:

- **Frame skips**: 0 (any skips are failures)
- **Duplicate frames**: 0 (any duplicates are failures)
- **FPS variance**: Standard deviation ≤ 1.0 across cameras
- **Minimum FPS**: ≥ 28.0 FPS
- **Timestamp spread**: ≤ 30ms (~1 frame at 30fps)
- **Zero timestamps**: 0 (any zero timestamps are failures)

## Example Output

```
TEST MATRIX
Cams    Frames   Reps   Avg FPS      Avg Spread (ms)   Skips    Duplicates
6       500      3      29.850       12.450            0        0
12      1000     3      29.920       18.320            0        0
```

## Related

- See acquisition system testing documentation for how to generate test results
