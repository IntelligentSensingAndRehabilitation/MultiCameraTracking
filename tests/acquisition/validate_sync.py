"""Pre-recording sync validation. Run via: make validate-sync CONFIG=/configs/your_config.yaml"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from multi_camera.acquisition.flir_recording_api import FlirRecorder

CONFIG_DIR = Path("/configs")


def list_available_configs() -> list[str]:
    if not CONFIG_DIR.exists():
        return []
    return [f.name for f in CONFIG_DIR.iterdir() if f.suffix == ".yaml"]


async def main():
    parser = argparse.ArgumentParser(description="Pre-recording sync validation")
    parser.add_argument("--config", required=True, help="Camera config YAML path")
    parser.add_argument(
        "--frames", type=int, default=100, help="Number of validation frames"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        configs = list_available_configs()
        if configs:
            print(f"\nAvailable configs in {CONFIG_DIR}:")
            for c in configs:
                print(f"  {CONFIG_DIR / c}")
        sys.exit(1)

    recorder = FlirRecorder()

    print(f"Configuring cameras from {args.config}")
    await recorder.configure_cameras(config_file=args.config)

    print(f"Running sync validation ({args.frames} frames)...")
    result = recorder.validate_sync(n_frames=args.frames)

    print()
    print(json.dumps(result, indent=2))

    if result["passed"]:
        print("\nPASSED: Sync quality is good, safe to record.")
    else:
        print(
            f"\nFAILED: {result['timespread_alert_count']} frames exceeded threshold."
        )
        print(f"  Max timespread: {result['max_timespread_ms']:.3f} ms")


if __name__ == "__main__":
    asyncio.run(main())
