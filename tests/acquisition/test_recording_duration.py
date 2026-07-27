"""Tests that write_metadata_queue emits per-file duration (#25).

Feeds synthetic frames through the metadata thread function synchronously and
checks both records_queue emit sites: the mid-stream file rollover and the
final flush. No hardware — PySpin is stubbed before import.
"""

from __future__ import annotations

import datetime
import sys
import types
from queue import Queue


def _install_pyspin_stubs() -> None:
    for name in ("PySpin", "simple_pyspin"):
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        if name == "simple_pyspin":

            class _Camera:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("PySpin stub")

            stub.Camera = _Camera  # type: ignore[attr-defined]
            stub._SYSTEM = None  # type: ignore[attr-defined]
            stub.list_cameras = lambda: []  # type: ignore[attr-defined]
        sys.modules[name] = stub


_install_pyspin_stubs()

from multi_camera.acquisition.flir_recording_api import write_metadata_queue  # noqa: E402

SERIALS = ["CAM_A", "CAM_B"]
BASE_NS = 1_000_000_000_000
T0 = datetime.datetime(2026, 7, 16, 14, 30, 0)


def _frame(base_filename: str, local_time: datetime.datetime, index: int) -> dict:
    return {
        "base_filename": base_filename,
        "real_times": local_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "local_times": local_time,
        "timestamps": [BASE_NS + index * 33_000_000] * len(SERIALS),
        "frame_id": [1000 + index] * len(SERIALS),
        "camera_serials": list(SERIALS),
        "exposure_times": [15000] * len(SERIALS),
        "frame_rates_requested": [30] * len(SERIALS),
        "frame_rates_binning": [30] * len(SERIALS),
    }


def test_duration_emitted_at_rollover_and_final_flush(tmp_path):
    file_a = str(tmp_path / "trial_a")
    file_b = str(tmp_path / "trial_b")

    json_queue: Queue = Queue()
    records_queue: Queue = Queue()

    # File A: two frames spanning 2.0 s; file B: three frames spanning 5.0 s.
    # The first file-B frame triggers the rollover emit for file A.
    json_queue.put(_frame(file_a, T0, 0))
    json_queue.put(_frame(file_a, T0 + datetime.timedelta(seconds=2), 1))
    json_queue.put(_frame(file_b, T0 + datetime.timedelta(seconds=10), 2))
    json_queue.put(_frame(file_b, T0 + datetime.timedelta(seconds=12.5), 3))
    json_queue.put(_frame(file_b, T0 + datetime.timedelta(seconds=15), 4))
    json_queue.put(None)

    config_metadata = {
        "chunk_data": False,
        "camera_config_hash": "deadbeef",
        "camera_info": {},
        "meta_info": {},
        "system_info": {},
    }

    write_metadata_queue(json_queue, records_queue, file_a, config_metadata)

    record_a = records_queue.get_nowait()  # rollover emit site
    record_b = records_queue.get_nowait()  # final-flush emit site
    assert records_queue.empty()

    assert record_a["filename"] == file_a
    assert record_a["recording_timestamp"] == T0
    assert record_a["duration"] == 2.0

    assert record_b["filename"] == file_b
    assert record_b["recording_timestamp"] == T0 + datetime.timedelta(seconds=10)
    assert record_b["duration"] == 5.0
