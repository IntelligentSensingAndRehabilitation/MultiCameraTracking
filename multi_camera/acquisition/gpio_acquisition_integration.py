"""
GPIO edge timestamp recording for FlirRecorder via ExposureEndLineStatusAll chunk data.

Line0 state is embedded in every frame's chunk data as ExposureEndLineStatusAll,
which piggybacks on the existing frame transfer at zero extra GigE cost. No
polling thread, no extra camera register reads during acquisition.

The per-frame line_status_all array is written to the acquisition JSON alongside
the existing timestamps array. Edge detection (rising/falling) and PTP timestamp
conversion are performed in GPIOEdgeTrigger.make() during DataJoint population.

To enable SmartWheel GPIO sync, set line0 to 'SmartWheel' in the camera config YAML:

    gpio-settings:
      line0: 'SmartWheel'

ExposureEndLineStatusAll is automatically added to chunk_data when SmartWheel is
enabled — no need to add it manually.

Line0 (opto-isolated input, Pin 2 on the 6-pin GPIO connector) is used because
the SmartWheel TTL signal is 5 V, which exceeds the 3.6 V maximum input high
level of Line3 (non-isolated input, Pin 1). Line0 accepts up to 30 V. Pin 5
(Opto GND) is the corresponding ground.
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

    Only instantiated when line0 = 'SmartWheel' in the camera config. Actual
    edge detection uses ExposureEndLineStatusAll chunk data written to the
    acquisition JSON per-frame. No polling thread, no extra GigE traffic
    during acquisition.

    Edge timestamps are extracted from the JSON during DataJoint population
    in GPIOEdgeTrigger.make() in smartwheel_gpio_sync.py.
    """

    def __init__(self, camera):
        self._camera = camera
        self._enabled = camera is not None and _PYSPIN_AVAILABLE

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
