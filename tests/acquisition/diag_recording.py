"""Short recording with diagnostics_level=1. Run via: make diag-recording CONFIG=/configs/your_config.yaml"""

import argparse
import asyncio
import datetime
import os
import sys

from multi_camera.acquisition.flir_recording_api import FlirRecorder

CONFIG_PATH = "/configs/"


def list_available_configs() -> list[str]:
    try:
        return [f for f in os.listdir(CONFIG_PATH) if f.endswith(".yaml")]
    except FileNotFoundError:
        return []


async def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic recording with system monitor"
    )
    parser.add_argument("--config", required=True, help="Camera config YAML path")
    parser.add_argument(
        "--output-dir", default="/data", help="Output directory (container path)"
    )
    parser.add_argument(
        "--frames", type=int, default=500, help="Number of frames to record"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        configs = list_available_configs()
        if configs:
            print(f"\nAvailable configs in {CONFIG_PATH}:")
            for c in configs:
                print(f"  {CONFIG_PATH}{c}")
        sys.exit(1)

    recorder = FlirRecorder()

    print(f"Configuring cameras from {args.config}")
    await recorder.configure_cameras(config_file=args.config)

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    diag_dir = f"{args.output_dir}/diagnostics/{date_str}"
    os.makedirs(diag_dir, exist_ok=True)
    recording_path = f"{diag_dir}/diag_test_{time_str}"

    print(f"Recording {args.frames} frames with diagnostics_level=1")
    print(f"Output: {recording_path}")

    records = recorder.start_acquisition(
        recording_path=recording_path,
        max_frames=args.frames,
        diagnostics_level=1,
    )

    print(f"\nRecording complete. {len(records)} segment(s).")
    for r in records:
        print(f"  {r['filename']}: spread={r['timestamp_spread']:.3f} ms")

    print(f"\nAnalyze with: make diag-analyze DATA={diag_dir}")


asyncio.run(main())
