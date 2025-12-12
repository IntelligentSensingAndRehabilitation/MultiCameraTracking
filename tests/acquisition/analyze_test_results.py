#!/usr/bin/env python3
"""
Parse and analyze test_matrix_results.json from MultiCameraTracking acquisition tests.
Generates summary tables and statistics for easy viewing.

CONFIGURABLE THRESHOLDS (matching test_acquisition.py):
- SKIP_THRESHOLD_PER_CAM: 0.0 (any skips are failures)
- DUP_THRESHOLD_PER_CAM: 0.0 (any duplicates are failures)
- FPS_STD_THRESHOLD: 1.0 (FPS standard deviation across cameras)
- FPS_MIN_THRESHOLD: 28.0 (minimum acceptable FPS)
- SPREAD_THRESHOLD_MS: 30.0 (maximum timestamp spread in ms)
- Baseline anomaly detection: Also flags if >3x the median for that configuration
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
import statistics


def load_test_results(filepath: str = "test_matrix_results.json") -> Dict[str, Any]:
    """Load the test results JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def parse_test_name(test_name: str) -> tuple:
    """
    Parse test name like 'test_6_500_rep0' into (num_cams, max_frames, repetition).
    """
    parts = test_name.split('_')
    num_cams = int(parts[1])
    max_frames = int(parts[2])
    rep = int(parts[3].replace('rep', ''))
    return num_cams, max_frames, rep


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"{title.center(80)}")
    print('=' * 80)


def print_overall_summary(results: Dict[str, Any]):
    """Print overall test summary statistics."""
    print_header("OVERALL TEST SUMMARY")

    total_tests = len(results)

    # Group by configuration
    configs = defaultdict(list)
    for test_name in results.keys():
        num_cams, max_frames, _ = parse_test_name(test_name)
        configs[(num_cams, max_frames)].append(test_name)

    print(f"\nTotal test runs: {total_tests}")
    print(f"Number of configurations: {len(configs)}")
    print(f"\nConfigurations tested:")
    print(f"{'Cameras':<10} {'Frames':<10} {'Repetitions':<15}")
    print('-' * 35)

    for (num_cams, max_frames), tests in sorted(configs.items()):
        print(f"{num_cams:<10} {max_frames:<10} {len(tests):<15}")


def print_test_configuration_summary(results: Dict[str, Any]):
    """Print summary statistics grouped by test configuration."""
    print_header("TEST CONFIGURATION SUMMARY")

    # Group by configuration
    configs = defaultdict(list)
    for test_name, test_data in results.items():
        num_cams, max_frames, _ = parse_test_name(test_name)
        configs[(num_cams, max_frames)].append(test_data)

    # Print table header
    header = f"{'Cams':<6} {'Frames':<8} {'Reps':<6} {'Avg FPS':<12} {'Spread (ms)':<15} {'Skips':<10} {'Dups':<10} {'Zero Counts':<12}"
    print(f"\n{header}")
    print('-' * len(header))

    for (num_cams, max_frames), tests in sorted(configs.items()):
        # Calculate statistics across repetitions
        fps_values = []
        spread_values = []
        total_skips = 0
        total_dups = 0
        total_zeros = 0

        for test in tests:
            # Get FPS values (average across cameras)
            cam_fps = list(test['timestamp_metrics']['raw_fps'].values())
            if cam_fps:
                fps_values.append(statistics.mean(cam_fps))

            # Get timestamp spread
            spread_values.append(test.get('timestamp_spread', 0))

            # Count skips
            total_skips += sum(test['frame_id_skips'].values())

            # Count duplicates
            for cam_dups in test['frame_id_duplicates'].values():
                total_dups += len(cam_dups)

            # Count zero timestamps
            total_zeros += test['timestamp_metrics'].get('zero_counts', 0)

        avg_fps = statistics.mean(fps_values) if fps_values else 0
        avg_spread = statistics.mean(spread_values) if spread_values else 0

        print(f"{num_cams:<6} {max_frames:<8} {len(tests):<6} "
              f"{avg_fps:<12.3f} {avg_spread * 1000:<15.3f} "
              f"{total_skips:<10} {total_dups:<10} {total_zeros:<12}")


def print_detailed_results(results: Dict[str, Any], config_filter: tuple = None):
    """Print detailed results for each test."""
    print_header("DETAILED TEST RESULTS")

    for test_name, test_data in sorted(results.items()):
        num_cams, max_frames, rep = parse_test_name(test_name)

        # Apply filter if specified
        if config_filter and (num_cams, max_frames) != config_filter:
            continue

        print(f"\n{test_name}")
        print('-' * 80)
        print(f"Configuration: {num_cams} cameras, {max_frames} frames, repetition {rep}")
        print(f"Recording timestamp: {test_data.get('recording_timestamp', 'N/A')}")
        print(f"Timestamp spread: {test_data.get('timestamp_spread', 0) * 1000:.3f} ms")
        print(f"Zero timestamp counts: {test_data['timestamp_metrics'].get('zero_counts', 0)}")

        # Camera-specific metrics
        print(f"\n{'Camera ID':<15} {'FPS':<10} {'Skips':<10} {'Duplicates':<12}")
        print('-' * 47)

        for cam_id in sorted(test_data['timestamp_metrics']['raw_fps'].keys()):
            fps = test_data['timestamp_metrics']['raw_fps'].get(cam_id, 0)
            skips = test_data['frame_id_skips'].get(cam_id, 0)
            dups = len(test_data['frame_id_duplicates'].get(cam_id, {}))
            print(f"{cam_id:<15} {fps:<10.3f} {skips:<10} {dups:<12}")


def evaluate_test_result(test_data: Dict[str, Any], num_cams: int, max_frames: int, baseline_spread: float = None) -> tuple[bool, list, dict]:
    """
    Evaluate a single test result against pass/fail criteria.

    Args:
        test_data: Test data dictionary
        num_cams: Number of cameras
        max_frames: Number of frames
        baseline_spread: Optional baseline timestamp spread (ms) for this configuration

    Returns:
        - passed (bool): Whether the test passed all criteria
        - problems (list): List of issue descriptions
        - metrics (dict): Key metrics for this test
    """
    problems = []

    # Check for frame skips
    total_skips = sum(test_data['frame_id_skips'].values())
    skips_per_cam = total_skips / num_cams if num_cams > 0 else 0

    # Check for duplicates
    total_dups = sum(len(dups) for dups in test_data['frame_id_duplicates'].values())
    dups_per_cam = total_dups / num_cams if num_cams > 0 else 0

    # Check for zero timestamps
    zero_counts = test_data['timestamp_metrics'].get('zero_counts', 0)

    # Check timestamp spread (already in ms from test_acquisition.py line 295)
    timestamp_spread_ms = test_data.get('timestamp_spread', 0)

    # Check for unusual FPS variance
    fps_values = list(test_data['timestamp_metrics']['raw_fps'].values())
    fps_std = statistics.stdev(fps_values) if len(fps_values) > 1 else 0
    fps_mean = statistics.mean(fps_values) if fps_values else 0
    fps_min = min(fps_values) if fps_values else 0
    fps_max = max(fps_values) if fps_values else 0

    # Thresholds for pass/fail - matching test_acquisition.py
    SKIP_THRESHOLD_PER_CAM = 0.0  # Any skips are failures (line 368 in test_acquisition.py)
    DUP_THRESHOLD_PER_CAM = 0.0   # Any duplicates are failures (line 372 in test_acquisition.py)
    FPS_STD_THRESHOLD = 1.0       # FPS std dev > 1.0
    FPS_MIN_THRESHOLD = 28.0      # Minimum acceptable FPS (line 353 in test_acquisition.py)

    # Timestamp spread threshold: Based on acquisition test threshold (line 349 in test_acquisition.py)
    # At 30fps: 1 frame = 33.33ms, but test uses 30ms threshold
    SPREAD_THRESHOLD_MS = 30.0  # Must match threshold in test_acquisition.py

    # Check for anomalous spread
    spread_is_anomaly = False

    # First check if spread exceeds the threshold (1 frame)
    if timestamp_spread_ms > SPREAD_THRESHOLD_MS:
        spread_is_anomaly = True
        frames_of_drift = timestamp_spread_ms / 33.33
        problems.append(f"high timestamp spread ({timestamp_spread_ms:.1f} ms = {frames_of_drift:.1f} frames)")

    # Additionally flag if it's an outlier compared to baseline (>3x)
    if baseline_spread is not None and baseline_spread > 0:
        if timestamp_spread_ms > baseline_spread * 3:
            if not spread_is_anomaly:  # Don't duplicate the message
                spread_is_anomaly = True
            problems.append(f"anomalous vs baseline ({timestamp_spread_ms:.1f} ms vs baseline {baseline_spread:.1f} ms)")

    # Evaluate criteria
    if total_skips > 0:
        problems.append(f"{total_skips} frame skips ({skips_per_cam:.2f} avg/camera)")

    if total_dups > 0:
        problems.append(f"{total_dups} duplicate frames ({dups_per_cam:.2f} avg/camera)")

    if zero_counts > 0:
        problems.append(f"{zero_counts} zero timestamps")

    if fps_std > FPS_STD_THRESHOLD:
        problems.append(f"high FPS variance (std={fps_std:.2f}, range={fps_min:.1f}-{fps_max:.1f})")

    if fps_min < FPS_MIN_THRESHOLD:
        problems.append(f"low FPS detected (min={fps_min:.1f} < {FPS_MIN_THRESHOLD})")

    # Determine pass/fail
    passed = (
        skips_per_cam <= SKIP_THRESHOLD_PER_CAM and
        dups_per_cam <= DUP_THRESHOLD_PER_CAM and
        zero_counts == 0 and
        not spread_is_anomaly and
        fps_std <= FPS_STD_THRESHOLD and
        fps_min >= FPS_MIN_THRESHOLD
    )

    metrics = {
        'skips_per_cam': skips_per_cam,
        'dups_per_cam': dups_per_cam,
        'spread_ms': timestamp_spread_ms,
        'fps_mean': fps_mean,
        'fps_std': fps_std,
    }

    return passed, problems, metrics


def calculate_baseline_spreads(results: Dict[str, Any]) -> Dict[tuple, float]:
    """Calculate baseline timestamp spreads for each configuration."""
    spreads_by_config = defaultdict(list)

    for test_name, test_data in results.items():
        num_cams, max_frames, _ = parse_test_name(test_name)
        spread_ms = test_data.get('timestamp_spread', 0)  # Already in ms
        spreads_by_config[(num_cams, max_frames)].append(spread_ms)

    # Use median as baseline to be robust against outliers
    baselines = {}
    for config, spreads in spreads_by_config.items():
        baselines[config] = statistics.median(spreads)

    return baselines


def print_test_matrix(results: Dict[str, Any]):
    """Print a matrix view of test results similar to the user's table."""
    print_header("TEST MATRIX")

    # Calculate baselines first
    baselines = calculate_baseline_spreads(results)

    # Get all unique camera counts and frame counts
    cam_counts = set()
    frame_counts = set()

    for test_name in results.keys():
        num_cams, max_frames, _ = parse_test_name(test_name)
        cam_counts.add(num_cams)
        frame_counts.add(max_frames)

    cam_counts = sorted(cam_counts)
    frame_counts = sorted(frame_counts)

    # Build matrix data
    matrix_data = {}
    for num_cams in cam_counts:
        for max_frames in frame_counts:
            # Get all repetitions for this config
            config_tests = []
            for test_name, test_data in results.items():
                tc, tf, _ = parse_test_name(test_name)
                if tc == num_cams and tf == max_frames:
                    config_tests.append((test_name, test_data))

            if config_tests:
                # Evaluate each repetition
                passes = 0
                total_reps = len(config_tests)
                all_skips = []
                all_dups = []
                all_problems = []

                baseline_spread = baselines.get((num_cams, max_frames))

                for test_name, test_data in config_tests:
                    passed, problems, metrics = evaluate_test_result(test_data, num_cams, max_frames, baseline_spread)
                    if passed:
                        passes += 1
                    all_skips.append(metrics['skips_per_cam'])
                    all_dups.append(metrics['dups_per_cam'])
                    if problems:
                        all_problems.extend(problems)

                avg_skips = statistics.mean(all_skips)
                avg_dups = statistics.mean(all_dups)

                matrix_data[(num_cams, max_frames)] = {
                    'passes': passes,
                    'total': total_reps,
                    'avg_skips': avg_skips,
                    'avg_dups': avg_dups,
                    'problems': all_problems
                }

    # Print matrix
    print(f"\n{'':>12}", end='')
    for num_cams in cam_counts:
        print(f"  {num_cams} cams{'':<12}", end='')
    print()

    print(f"{'# frames':<12}", end='')
    print('-' * (len(cam_counts) * 20))

    for max_frames in frame_counts:
        print(f"{max_frames:<12}", end='')
        for num_cams in cam_counts:
            if (num_cams, max_frames) in matrix_data:
                data = matrix_data[(num_cams, max_frames)]
                passes = data['passes']
                total = data['total']

                if passes == total:
                    # All passed
                    cell = f"passed {passes}/{total}"
                elif data['avg_skips'] > 0:
                    # Has skips
                    cell = f"{passes}/{total} ({data['avg_skips']:.2f} skips/cam)"
                elif data['avg_dups'] > 0:
                    # Has duplicates
                    cell = f"{passes}/{total} ({data['avg_dups']:.2f} dups/cam)"
                else:
                    # Failed for other reasons
                    cell = f"{passes}/{total}"

                print(f"  {cell:<18}", end='')
            else:
                print(f"  {'-':<18}", end='')
        print()

    print("\nLegend: passed X/Y = X tests passed out of Y total repetitions")
    print("        Values in parentheses show average per-camera metrics")


def print_failure_analysis(results: Dict[str, Any]):
    """Identify and print any tests with issues."""
    print_header("FAILURE ANALYSIS")

    # Calculate baselines
    baselines = calculate_baseline_spreads(results)

    issues_found = False

    for test_name, test_data in sorted(results.items()):
        num_cams, max_frames, _ = parse_test_name(test_name)
        baseline_spread = baselines.get((num_cams, max_frames))
        passed, problems, metrics = evaluate_test_result(test_data, num_cams, max_frames, baseline_spread)

        if not passed:
            issues_found = True
            print(f"\n{test_name} (FAILED):")
            for problem in problems:
                print(f"  - {problem}")

            # Show detailed camera info if there are FPS issues
            if metrics['fps_std'] > 1.0:
                print(f"  Camera FPS details:")
                for cam_id, fps in sorted(test_data['timestamp_metrics']['raw_fps'].items()):
                    print(f"    {cam_id}: {fps:.3f} FPS")

    if not issues_found:
        print("\nNo issues detected! All tests passed successfully.")


def print_fps_comparison(results: Dict[str, Any]):
    """Print FPS comparison across different camera counts."""
    print_header("FPS COMPARISON BY CAMERA COUNT")

    # Group by number of cameras
    fps_by_cam_count = defaultdict(list)

    for test_name, test_data in results.items():
        num_cams, _, _ = parse_test_name(test_name)
        cam_fps = list(test_data['timestamp_metrics']['raw_fps'].values())
        if cam_fps:
            fps_by_cam_count[num_cams].extend(cam_fps)

    print(f"\n{'Cameras':<10} {'Count':<10} {'Mean FPS':<12} {'Min FPS':<12} {'Max FPS':<12} {'Std Dev':<12}")
    print('-' * 68)

    for num_cams in sorted(fps_by_cam_count.keys()):
        fps_values = fps_by_cam_count[num_cams]
        mean_fps = statistics.mean(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)
        std_fps = statistics.stdev(fps_values) if len(fps_values) > 1 else 0

        print(f"{num_cams:<10} {len(fps_values):<10} {mean_fps:<12.3f} "
              f"{min_fps:<12.3f} {max_fps:<12.3f} {std_fps:<12.3f}")


def main():
    """Main entry point."""
    # Check for input file
    input_file = sys.argv[1] if len(sys.argv) > 1 else "test_matrix_results.json"

    if not Path(input_file).exists():
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Load results
    print(f"Loading test results from: {input_file}")
    results = load_test_results(input_file)

    # Print various summaries
    print_overall_summary(results)
    print_test_matrix(results)
    print_test_configuration_summary(results)
    print_fps_comparison(results)
    print_failure_analysis(results)

    # Optional: Print detailed results for a specific configuration
    # Uncomment and modify to see detailed results for specific tests
    # print_detailed_results(results, config_filter=(6, 500))

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
