"""
GPIO edge timestamp recording for FlirRecorder via PySpin hardware events.

Records one rising edge (SmartWheel start) and one falling edge (SmartWheel
stop) on a single camera's Line0 GPIO input using the camera's built-in
event system. Timestamps are in the camera's PTP timebase — the same domain
as frame timestamps — giving sub-frame precision alignment without polling.

Line0 (opto-isolated input, Pin 2 on the 6-pin GPIO connector) is used
because the SmartWheel TTL signal is 5 V, which exceeds the 3.6 V maximum
input high level of Line3 (non-isolated input, Pin 1). Line0 accepts up to
30 V and is designed for interfacing with external systems at various voltage
levels. Pin 5 (Opto GND) is the corresponding ground.

When line0 = "Off" in the camera config YAML (the default for PTP-triggered
recording), init_camera() sets TriggerSource = "Action0" and does not touch
Line0 at all, so Line0 stays at its power-on default (input, events disabled)
— no config change is needed.

IMPORTANT: Do not use this with line0 = "ArduinoTrigger". In that mode Line0
is already the per-frame trigger source; registering edge events on it would
fire on every frame. The GPIOEdgeRecorder constructor accepts a
line0_used_for_trigger flag and silently disables itself when True.

Integration into FlirRecorder (flir_recording_api.py)
------------------------------------------------------
After configure_cameras() has been called and self.cams is populated, create
the recorder once and keep it for the lifetime of the FlirRecorder:

    from gpio_acquisition_integration import GPIOEdgeRecorder
    # Pick any one camera; Line0 on that camera receives the SmartWheel TTL.
    line0_is_trigger = self.gpio_settings.get("line0") == "ArduinoTrigger"
    self.gpio_recorder = GPIOEdgeRecorder(self.cams[0], line0_used_for_trigger=line0_is_trigger)

In start_acquisition(), after cameras have been started (after the
start_cam executor block, ~line 1161):

    self.gpio_recorder.start()

In start_acquisition(), just before json_queue.put(None) (~line 1683):

    gpio_data = self.gpio_recorder.stop()

After json_queue.join(), patch the result into the written JSON:

    if gpio_data and self.video_base_file is not None:
        import json, pathlib
        p = pathlib.Path(self.video_base_file + ".json")
        obj = json.loads(p.read_text())
        obj["gpio_line_0"] = gpio_data
        p.write_text(json.dumps(obj) + "\n")
"""

import threading

try:
    import PySpin
    _PYSPIN_AVAILABLE = True
except ImportError:
    _PYSPIN_AVAILABLE = False


class _Line0EventHandler(PySpin.DeviceEventHandler if _PYSPIN_AVAILABLE else object):
    """
    PySpin device event handler that records Line0 rising and falling edges.

    OnDeviceEvent is called on the PySpin internal callback thread. It reads
    GetTimestamp() which returns nanoseconds in the camera's PTP timebase.
    The lock protects self.edges from concurrent reads in the main thread.
    """

    def __init__(self):
        if _PYSPIN_AVAILABLE:
            super().__init__()
        self.edges: list[tuple[float, str]] = []
        self.lock = threading.Lock()

    def OnDeviceEvent(self, event_name):
        ts_ns = self.GetTimestamp()
        ts_s = ts_ns / 1e9
        name = str(event_name)
        if "Rising" in name:
            edge_type = "rising"
        elif "Falling" in name:
            edge_type = "falling"
        else:
            return
        with self.lock:
            self.edges.append((ts_s, edge_type))


class GPIOEdgeRecorder:
    """
    Monitor Line0 on one FLIR camera for exactly one rising and one falling
    edge using PySpin hardware GPIO events.

    Line0 is the opto-isolated input (Pin 2, Opto GND on Pin 5). It accepts
    signals up to 30 V, making it safe for the SmartWheel's 5 V TTL output.

    The camera's onboard event engine detects the edge in hardware and
    delivers a PTP-synchronized timestamp via the DeviceEventHandler
    callback. No polling. No host OS GPIO involvement.

    Timestamps are in nanoseconds-since-epoch in the camera's PTP clock,
    the same domain as im_ref.GetTimeStamp() frame timestamps. They can be
    compared directly to the per-frame timestamps in the acquisition JSON.

    Do not use when line0 = "ArduinoTrigger" — pass line0_used_for_trigger=True
    and the recorder will silently disable itself.
    """

    # EventSelector values (no "Event" prefix — GenICam selector enum strings)
    _RISING_SELECTOR  = "Line0RisingEdge"
    _FALLING_SELECTOR = "Line0FallingEdge"
    # Full event names used with RegisterEventHandler and in the callback
    _RISING_EVENT  = "EventLine0RisingEdge"
    _FALLING_EVENT = "EventLine0FallingEdge"

    def __init__(self, camera, line0_used_for_trigger: bool = False):
        """
        Args:
            camera: An initialised PySpin Camera object (init_camera() has
                    already been called on it). The camera does not need to
                    be streaming — events work before and during acquisition.
                    Pass None to silently disable edge recording (no crash).
            line0_used_for_trigger: Set True when line0 = "ArduinoTrigger" in
                    the camera config. Disables edge recording to avoid
                    spurious events on every frame trigger.
        """
        self._camera = camera
        self._handler: _Line0EventHandler | None = None
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
        Configure Line0 as an input and register PySpin event handlers for
        rising and falling edges. Safe to call while the camera is streaming.
        """
        if not self._enabled:
            return
        try:
            c = self._camera

            # Debug: dump all top-level event-related nodes and their values
            import PySpin
            node_map = c.cam.GetNodeMap()
            print("=== Event-related nodes ===")
            for node_name in [
                "EventEnable", "AcquisitionStatusSelector",
                "TransferEventEnable", "GevSCPSDoNotFragment",
                "EventExposureEndFrameID",
            ]:
                try:
                    node = node_map.GetNode(node_name)
                    if PySpin.IsAvailable(node):
                        print(f"  {node_name}: available")
                    else:
                        print(f"  {node_name}: not available")
                except Exception as e:
                    print(f"  {node_name}: error - {e}")

            # Try enabling event engine first, then query EventSelector
            print("=== Trying to enable event engine ===")
            for enable_node in ["EventEnable", "AcquisitionEventEnable"]:
                try:
                    node = PySpin.CBooleanPtr(node_map.GetNode(enable_node))
                    if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                        node.SetValue(True)
                        print(f"  Set {enable_node} = True")
                except Exception as e:
                    print(f"  {enable_node}: {e}")

            # Now retry EventSelector
            print("=== EventSelector entries after enable attempt ===")
            try:
                selector = PySpin.CEnumerationPtr(node_map.GetNode('EventSelector'))
                entries = selector.GetEntries()
                for e in entries:
                    print(f"  {PySpin.CEnumEntryPtr(e).GetSymbolic()}")
            except Exception as e:
                print(f"  EventSelector query failed: {e}")

            # Ensure Line0 is in input mode. When line0 = "Off" in the camera
            # config yaml, init_camera() does not touch Line0, so this sets
            # it explicitly to input before enabling events.
            c.LineSelector = "Line0"
            c.LineMode = "Input"

            # Enable edge-detection events on Line0.
            # EventSelector uses the selector string (no "Event" prefix);
            # RegisterEventHandler uses the full event name (with prefix).
            c.EventSelector = self._RISING_SELECTOR
            c.EventNotification = "On"
            c.EventSelector = self._FALLING_SELECTOR
            c.EventNotification = "On"

            self._handler = _Line0EventHandler()
            c.cam.RegisterEventHandler(self._handler, self._RISING_EVENT)
            c.cam.RegisterEventHandler(self._handler, self._FALLING_EVENT)

        except Exception as exc:
            print(f"GPIOEdgeRecorder: failed to start — {exc}. Edge recording disabled.")
            self._enabled = False
            self._handler = None

    def stop(self) -> dict:
        """
        Unregister event handlers and return edge data for JSON storage.

        Returns a dict with keys ptp_times, edge_types, rising_time, and
        falling_time if exactly one rising and one falling edge were recorded,
        otherwise returns an empty dict with a printed warning.

        Call this after stopping acquisition but before closing the JSON file.
        """
        if not self._enabled or self._handler is None:
            return {}

        try:
            c = self._camera
            c.cam.UnregisterEventHandler(self._handler, self._RISING_EVENT)
            c.cam.UnregisterEventHandler(self._handler, self._FALLING_EVENT)
            c.EventSelector = self._RISING_SELECTOR
            c.EventNotification = "Off"
            c.EventSelector = self._FALLING_SELECTOR
            c.EventNotification = "Off"
        except Exception as exc:
            print(f"GPIOEdgeRecorder: error during stop — {exc}")

        with self._handler.lock:
            edges = list(self._handler.edges)

        if len(edges) != 2:
            print(f"GPIOEdgeRecorder: expected 2 edges (rising + falling), got {len(edges)}")
            return {}

        ptp_times  = [e[0] for e in edges]
        edge_types = [e[1] for e in edges]

        rising_times  = [t for t, k in zip(ptp_times, edge_types) if k == "rising"]
        falling_times = [t for t, k in zip(ptp_times, edge_types) if k == "falling"]

        if len(rising_times) != 1 or len(falling_times) != 1:
            print(
                f"GPIOEdgeRecorder: expected 1 rising + 1 falling, "
                f"got {len(rising_times)} rising / {len(falling_times)} falling"
            )
            return {}

        return {
            "ptp_times":   ptp_times,
            "edge_types":  edge_types,
            "rising_time": float(rising_times[0]),
            "falling_time": float(falling_times[0]),
        }
