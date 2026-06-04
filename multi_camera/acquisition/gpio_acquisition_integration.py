"""
GPIO edge timestamp recording for FlirRecorder via ExposureEnd events.

The BFS-PGE-31S4C firmware does not expose Line0 edge events through the
PySpin EventSelector system. Instead, this module uses ExposureEnd events
(which ARE supported) to sample Line0 state at the end of every frame
exposure. Rising and falling edges are detected by comparing the Line0
state across consecutive frames, and the PTP timestamp of the ExposureEnd
event is used as the edge timestamp.

Precision is one frame period (~33 ms at 30 fps). For SmartWheel alignment
at 240 Hz this means ~8 samples of uncertainty on the start/stop boundary,
which is acceptable for most biomechanics analyses.

Line0 (opto-isolated input, Pin 2 on the 6-pin GPIO connector) is used
because the SmartWheel TTL signal is 5 V, which exceeds the 3.6 V maximum
input high level of Line3 (non-isolated input, Pin 1). Line0 accepts up to
30 V and is designed for interfacing with external systems at various voltage
levels. Pin 5 (Opto GND) is the corresponding ground.

IMPORTANT: Do not use this with line0 = "ArduinoTrigger". In that mode Line0
is already the per-frame trigger source. The GPIOEdgeRecorder constructor
accepts a line0_used_for_trigger flag and silently disables itself when True.

Integration into FlirRecorder (flir_recording_api.py)
------------------------------------------------------
After configure_cameras() has been called and self.cams is populated, create
the recorder once and keep it for the lifetime of the FlirRecorder:

    from gpio_acquisition_integration import GPIOEdgeRecorder
    line0_is_trigger = self.gpio_settings.get("line0") == "ArduinoTrigger"
    self.gpio_recorder = GPIOEdgeRecorder(self.cams[0], line0_used_for_trigger=line0_is_trigger)

In start_acquisition(), BEFORE the start_cam executor block, guarded by
recording_path is not None:

    if self.gpio_recorder is not None and recording_path is not None:
        self.gpio_recorder.start()

In start_acquisition(), in the teardown section, guarded by recording_path
is not None:

    gpio_data = self.gpio_recorder.stop() if self.gpio_recorder is not None and recording_path is not None else {}

After json_queue.join(), patch the result into the written JSON:

    if gpio_data and self.video_base_file is not None:
        import json, pathlib
        p = pathlib.Path(self.video_base_file + ".json")
        obj = json.loads(p.read_text())
        obj["gpio_line_0"] = gpio_data
        p.write_text(json.dumps(obj) + "\\n")
"""

import threading

try:
    import PySpin
    _PYSPIN_AVAILABLE = True
except ImportError:
    _PYSPIN_AVAILABLE = False


class _ExposureEndHandler(PySpin.DeviceEventHandler if _PYSPIN_AVAILABLE else object):
    """
    PySpin ExposureEnd event handler that detects Line0 edges by sampling
    LineStatusAll at the end of every frame exposure.

    OnDeviceEvent fires on the PySpin internal callback thread after each
    frame exposure completes. GetTimestamp() returns nanoseconds in the
    camera's PTP timebase — the same domain as frame timestamps.

    Edge detection: compare Line0 bit in LineStatusAll across consecutive
    callbacks. Record the PTP timestamp of the first low→high transition
    (rising edge) and the first high→low transition (falling edge).
    """

    # Line0 is bit 0 of LineStatusAll
    _LINE0_BIT = 0x1

    def __init__(self):
        if _PYSPIN_AVAILABLE:
            super().__init__()
        self.edges: list[tuple[float, str]] = []
        self.lock = threading.Lock()
        self._prev_line0: bool | None = None
        self._cam_ref = None  # set by GPIOEdgeRecorder after construction

    def OnDeviceEvent(self, event_name):
        if self._cam_ref is None:
            return
        try:
            import time
            ts_s = time.time()
            line0_high = bool(self._cam_ref.LineStatusAll & self._LINE0_BIT)

            with self.lock:
                if self._prev_line0 is None:
                    self._prev_line0 = line0_high
                    return

                if not self._prev_line0 and line0_high:
                    self.edges.append((ts_s, "rising"))
                    print(f"GPIOEdgeRecorder: rising edge detected at {ts_s:.6f} s")
                elif self._prev_line0 and not line0_high:
                    self.edges.append((ts_s, "falling"))
                    print(f"GPIOEdgeRecorder: falling edge detected at {ts_s:.6f} s")

                self._prev_line0 = line0_high
        except Exception as exc:
            print(f"GPIOEdgeRecorder: error in ExposureEnd callback — {exc}")


class GPIOEdgeRecorder:
    """
    Monitor Line0 on one FLIR camera for rising and falling edges by sampling
    LineStatusAll on every ExposureEnd event.

    The BFS-PGE-31S4C does not support Line0 edge events in EventSelector,
    so ExposureEnd events are used instead. Line0 state is sampled at the
    end of each frame exposure (~33 ms precision at 30 fps).

    Line0 is the opto-isolated input (Pin 2, Opto GND on Pin 5). It accepts
    signals up to 30 V, making it safe for the SmartWheel's 5 V TTL output.

    Timestamps are in the camera's PTP timebase (seconds since epoch),
    the same domain as frame timestamps in the acquisition JSON.

    Do not use when line0 = "ArduinoTrigger" — pass line0_used_for_trigger=True
    and the recorder will silently disable itself.
    """

    _EXPOSURE_END_SELECTOR = "ExposureEnd"
    _EXPOSURE_END_EVENT    = "EventExposureEnd"

    def __init__(self, camera, line0_used_for_trigger: bool = False):
        """
        Args:
            camera: An initialised simple_pyspin Camera object. Pass None to
                    silently disable edge recording (no crash).
            line0_used_for_trigger: Set True when line0 = "ArduinoTrigger" in
                    the camera config. Disables edge recording.
        """
        self._camera = camera
        self._handler: _ExposureEndHandler | None = None
        self._enabled = (
            camera is not None
            and _PYSPIN_AVAILABLE
            and not line0_used_for_trigger
        )
        if line0_used_for_trigger:
            print(
                "GPIOEdgeRecorder: line0 is configured as ArduinoTrigger — "
                "SmartWheel GPIO edge recording disabled."
            )

    def start(self) -> None:
        """
        Enable ExposureEnd events and register the handler. Call this after
        cameras have started streaming and recording_path is not None.
        """
        if not self._enabled:
            return
        try:
            c = self._camera

            c.LineSelector = "Line0"
            c.LineMode = "Input"

            c.EventSelector = self._EXPOSURE_END_SELECTOR
            c.EventNotification = "On"

            self._handler = _ExposureEndHandler()
            self._handler._cam_ref = c
            c.cam.RegisterEventHandler(self._handler, self._EXPOSURE_END_EVENT)

            print("GPIOEdgeRecorder: started — sampling Line0 on ExposureEnd events")

        except Exception as exc:
            print(f"GPIOEdgeRecorder: failed to start — {exc}. Edge recording disabled.")
            self._enabled = False
            self._handler = None

    def stop(self) -> dict:
        """
        Unregister the ExposureEnd handler and return edge data for JSON storage.

        Returns a dict with keys ptp_times, edge_types, rising_time, and
        falling_time if at least one rising and one falling edge were recorded.
        Returns an empty dict with a printed warning otherwise.
        """
        if not self._enabled or self._handler is None:
            return {}

        try:
            c = self._camera
            c.cam.UnregisterEventHandler(self._handler, self._EXPOSURE_END_EVENT)
            c.EventSelector = self._EXPOSURE_END_SELECTOR
            c.EventNotification = "Off"
        except Exception as exc:
            print(f"GPIOEdgeRecorder: error during stop — {exc}")

        with self._handler.lock:
            edges = list(self._handler.edges)

        print(f"GPIOEdgeRecorder: stop — recorded {len(edges)} edge(s): {edges}")

        ptp_times  = [e[0] for e in edges]
        edge_types = [e[1] for e in edges]

        rising_times  = [t for t, k in zip(ptp_times, edge_types) if k == "rising"]
        falling_times = [t for t, k in zip(ptp_times, edge_types) if k == "falling"]

        if len(rising_times) < 1 or len(falling_times) < 1:
            print(
                f"GPIOEdgeRecorder: expected at least 1 rising + 1 falling edge, "
                f"got {len(rising_times)} rising / {len(falling_times)} falling"
            )
            return {}

        # Use first rising and last falling edge
        rising_time  = float(rising_times[0])
        falling_time = float(falling_times[-1])

        if falling_time <= rising_time:
            print("GPIOEdgeRecorder: falling edge not after rising edge — discarding")
            return {}

        return {
            "ptp_times":   ptp_times,
            "edge_types":  edge_types,
            "rising_time": rising_time,
            "falling_time": falling_time,
        }
