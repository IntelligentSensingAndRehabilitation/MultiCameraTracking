"""
GPIO edge timestamp recording for FlirRecorder via background polling thread.

The BFS-PGE-31S4C firmware does not expose Line0 edge events through the
PySpin EventSelector system. A background thread polls LineStatusAll at
20 Hz (every 50 ms) to detect rising and falling edges on Line0 without
interfering with frame acquisition.

Precision is ~50 ms (one poll interval). For SmartWheel alignment at 240 Hz
this means ~12 samples of uncertainty on the start/stop boundary, which is
acceptable for most biomechanics analyses.

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
        import pathlib
        p = pathlib.Path(self.video_base_file + ".json")
        obj = json.loads(p.read_text())
        obj["gpio_line_0"] = gpio_data
        p.write_text(json.dumps(obj) + "\\n")
"""

import threading
import time

try:
    import PySpin
    _PYSPIN_AVAILABLE = True
except ImportError:
    _PYSPIN_AVAILABLE = False


class _Line0PollingThread(threading.Thread):
    """
    Background thread that polls Line0 state at 20 Hz (every 50 ms).

    Reads LineStatusAll from the camera at a low fixed rate to detect
    rising and falling edges on Line0. Running this in a separate thread
    at low frequency avoids interfering with the PySpin acquisition loop.

    Timestamps are host wall clock (time.time()) and are converted to PTP
    during DataJoint population using the frame timestamp mapping.
    """

    _LINE0_BIT = 0x1
    _POLL_INTERVAL_S = 0.05  # 20 Hz

    def __init__(self, camera):
        super().__init__(name="gpio_line0_poll", daemon=True)
        self._camera = camera
        self.edges: list[tuple[float, str]] = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._prev_line0: bool | None = None

    def run(self):
        while not self._stop_event.is_set():
            try:
                line0_high = bool(self._camera.LineStatusAll & self._LINE0_BIT)
                ts_s = time.time()

                with self.lock:
                    if self._prev_line0 is None:
                        self._prev_line0 = line0_high
                    elif not self._prev_line0 and line0_high:
                        self.edges.append((ts_s, "rising"))
                        print(f"GPIOEdgeRecorder: rising edge at {ts_s:.6f} s")
                        self._prev_line0 = line0_high
                    elif self._prev_line0 and not line0_high:
                        self.edges.append((ts_s, "falling"))
                        print(f"GPIOEdgeRecorder: falling edge at {ts_s:.6f} s")
                        self._prev_line0 = line0_high
                    else:
                        self._prev_line0 = line0_high

            except Exception as exc:
                print(f"GPIOEdgeRecorder: polling error — {exc}")

            self._stop_event.wait(self._POLL_INTERVAL_S)

    def stop(self):
        self._stop_event.set()


class GPIOEdgeRecorder:
    """
    Monitor Line0 on one FLIR camera for rising and falling edges using a
    low-frequency background polling thread (20 Hz).

    The BFS-PGE-31S4C does not support Line0 edge events in EventSelector,
    so LineStatusAll is polled at 20 Hz instead. This avoids interfering
    with frame acquisition by keeping GigE traffic minimal.

    Line0 is the opto-isolated input (Pin 2, Opto GND on Pin 5). It accepts
    signals up to 30 V, making it safe for the SmartWheel's 5 V TTL output.

    Timestamps are host wall clock time (time.time()). The DataJoint
    GPIOEdgeTrigger table converts them to PTP using the frame timestamp
    mapping from the acquisition JSON.

    Do not use when line0 = "ArduinoTrigger" — pass line0_used_for_trigger=True
    and the recorder will silently disable itself.
    """

    def __init__(self, camera, line0_used_for_trigger: bool = False):
        """
        Args:
            camera: An initialised simple_pyspin Camera object. Pass None to
                    silently disable edge recording (no crash).
            line0_used_for_trigger: Set True when line0 = "ArduinoTrigger" in
                    the camera config. Disables edge recording.
        """
        self._camera = camera
        self._poller: _Line0PollingThread | None = None
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
        Set Line0 to input mode and start the background polling thread.
        Call this after cameras have started streaming and only when
        recording_path is not None.
        """
        if not self._enabled:
            return
        try:
            c = self._camera
            c.LineSelector = "Line0"
            c.LineMode = "Input"

            self._poller = _Line0PollingThread(c)
            self._poller.start()
            print("GPIOEdgeRecorder: started — polling Line0 at 20 Hz")

        except Exception as exc:
            print(f"GPIOEdgeRecorder: failed to start — {exc}. Edge recording disabled.")
            self._enabled = False
            self._poller = None

    def stop(self) -> dict:
        """
        Stop the polling thread and return edge data for JSON storage.

        Returns a dict with keys host_times, edge_types, rising_time, and
        falling_time if at least one rising and one falling edge were recorded.
        host_times are in host wall clock seconds (time.time()) and will be
        converted to PTP during DataJoint population. Returns an empty dict
        with a printed warning otherwise.
        """
        if not self._enabled or self._poller is None:
            return {}

        self._poller.stop()
        self._poller.join(timeout=1.0)

        with self._poller.lock:
            edges = list(self._poller.edges)

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

        rising_time  = float(rising_times[0])
        falling_time = float(falling_times[-1])

        if falling_time <= rising_time:
            print("GPIOEdgeRecorder: falling edge not after rising edge — discarding")
            return {}

        return {
            "host_times":  ptp_times,
            "edge_types":  edge_types,
            "rising_time": rising_time,
            "falling_time": falling_time,
        }
