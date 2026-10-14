"""
GPIO constants for SmartWheel TTL synchronization via ExposureEndLineStatusAll chunk data.

When line0 = 'SmartWheel' in the camera config gpio-settings, FlirRecorder
automatically enables ExposureEndLineStatusAll chunk data on all cameras. The
per-frame LineStatusAll bitmask is stored in the acquisition JSON alongside
timestamps. Edge detection is performed in GPIOEdgeTrigger.make() during
DataJoint population.

Line0 (opto-isolated input, Pin 2 on the 6-pin GPIO connector) accepts up to
30 V. The SmartWheel TTL signal is 5 V. Pin 5 (Opto GND) is the ground.
"""

# Line0 is bit 0 of LineStatusAll / ExposureEndLineStatusAll
LINE0_BIT = 0x1
