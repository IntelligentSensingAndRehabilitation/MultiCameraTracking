#!/usr/bin/env python3
"""CLI tool to validate ArUco marker visibility across multi-camera recordings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from multi_camera.analysis.aruco_validation import (
    WalkwayMarkerConfig,
    compute_coverage_matrix,
    detect_markers_multi_camera,
    discover_camera_videos,
    generate_summary_report,
    make_aruco_grid_video,
    save_results_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ArUco marker visibility across camera views."
    )
    parser.add_argument(
        "video_base",
        help="Base filename for multi-camera recording (e.g. t111_20260113_114850)",
    )
    parser.add_argument(
        "--video-dir", required=True, help="Directory containing the .mp4 video files"
    )
    parser.add_argument(
        "--output-dir",
        default="./aruco_results",
        help="Directory for output files (default: ./aruco_results)",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Skip grid overlay video generation"
    )
    parser.add_argument(
        "--min-cameras",
        type=int,
        default=2,
        help="Minimum cameras required per marker (default: 2)",
    )
    parser.add_argument(
        "--downsample", type=int, default=2, help="Video downsample factor (default: 2)"
    )
    args = parser.parse_args()

    config = WalkwayMarkerConfig(min_cameras=args.min_cameras)

    print(f"Discovering videos for: {args.video_base}")
    video_paths = discover_camera_videos(args.video_base, args.video_dir)
    if not video_paths:
        print(f"No videos found matching '{args.video_base}.*.mp4' in {args.video_dir}")
        raise SystemExit(1)

    print(f"Found {len(video_paths)} cameras: {', '.join(sorted(video_paths.keys()))}")

    print("Detecting ArUco markers across all cameras...")
    results = detect_markers_multi_camera(video_paths, config)

    coverage = compute_coverage_matrix(results, config.all_marker_ids)
    report = generate_summary_report(coverage, config)
    print(report)

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = str(Path(args.output_dir) / f"{args.video_base}_aruco.json")
    save_results_json(results, coverage, config, json_path)
    print(f"Results saved to: {json_path}")

    if not args.no_video:
        video_out = str(Path(args.output_dir) / f"{args.video_base}_aruco_grid.mp4")
        print("Generating grid overlay video...")
        make_aruco_grid_video(
            video_paths, results, config, video_out, downsample=args.downsample
        )
        print(f"Grid video saved to: {video_out}")


if __name__ == "__main__":
    main()
