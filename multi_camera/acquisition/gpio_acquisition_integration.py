"""
GPIO edge timestamp recording for FlirRecorder via ExposureEndLineStatusAll chunk data.

Line0 state is embedded in every frame's chunk data as ExposureEndLineStatusAll,
which piggybacks on the existing frame transfer at zero extra GigE cost. No
polling thread, no extra camera register reads during acquisition.

The per-frame line_status_all array is written to the acquisition JSON alongside
the existing timestamps array. Edge detection (rising/falling) and PTP timestamp
conversion are performed in GPIOEdgeTrigger.make() during DataJoint population.

To use this, add "ExposureEndLineStatusAll" to the chunk_data list in the camera
config YAML:

    acquisition-settings:
      chunk_data:
        - FrameID
        - SerialData
        - ExposureEndLineStatusAll

Line0 (opto-isolated input, Pin 2 on the 6-pin GPIO connector) is used because
the SmartWheel TTL signal is 5 V, which exceeds the 3.6 V maximum input high
level of Line3 (non-isolated input, Pin 1). Line0 accepts up to 30 V. Pin 5
(Opto GND) is the corresponding ground.

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

GPIOEdgeRecorder.start() and stop() are no-ops — edge detection happens
in DataJoint. The recorder is kept for API compatibility and to log whether
GPIO recording is active. The recording_path guards in flir_recording_api.py
should still be kept to avoid calling start/stop during preview.
"""

try:
    import PySpin
    _PYSPIN_AVAILABLE = True
except ImportError:
    _PYSPIN_AVAILABLE = False

# Line0 is bit 0 of LineStatusAll / ExposureEndLineStatusAll
LINE0_BIT = 0x1


class GPIOEdgeRecorder:
    """
    Marker class that signals GPIO edge recording is active via chunk data.

    Actual edge detection uses ExposureEndLineStatusAll chunk data written
    to the acquisition JSON per-frame. No polling thread, no extra GigE
    traffic during acquisition.

    Edge timestamps are extracted from the JSON during DataJoint population
    in GPIOEdgeTrigger.make() in smartwheel_gpio_sync.py.

    Requires "ExposureEndLineStatusAll" in the camera config chunk_data list.
    """

    def __init__(self, camera, line0_used_for_trigger: bool = False):
        self._camera = camera
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
        if not self._enabled:
            return
        try:
            c = self._camera
            c.LineSelector = "Line0"
            c.LineMode = "Input"
            print(
                "GPIOEdgeRecorder: active — Line0 state recorded via "
                "ExposureEndLineStatusAll chunk data each frame"
            )
        except Exception as exc:
            print(f"GPIOEdgeRecorder: failed to configure Line0 — {exc}")
            self._enabled = False

    def stop(self) -> dict:
        if not self._enabled:
            return {}
        print(
            "GPIOEdgeRecorder: stop — edge timestamps will be extracted "
            "from line_status_all in the acquisition JSON during DataJoint population"
        )
        return {"chunk_based": True}
