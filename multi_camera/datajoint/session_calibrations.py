import datajoint as dj

from .calibrate_cameras import Calibration
from .multi_camera_dj import MultiCameraRecording, CalibratedRecording
from .sessions import Session, Recording

schema = dj.schema("multicamera_session_calibrations")

@schema
class SessionCalibration(dj.Computed):
    definition = """
    -> Session
    ---
    entry_time = CURRENT_TIMESTAMP : timestamp
    """

    class Grouping(dj.Part):
        definition = """
        -> master
        -> Calibration
        """

    class Recordings(dj.Part):
        definition = """
        -> SessionCalibration.Grouping
        -> Recording
        -> CalibratedRecording
        """

    def make(self, key):

        # get the recordings for this session
        recordings = (Recording & key).fetch("KEY")

        # use the multi_camera_recordings to get the calibrated recordings for this session
        calibrated_recordings = (CalibratedRecording & recordings).fetch("KEY")

        # get the calibrations from the calibrated recordings
        calibrations = (Calibration & calibrated_recordings).fetch("KEY")

        # insert the key into the SessionCalibration table
        self.insert1(key)

        # insert the calibrations into the Grouping table
        self.Grouping.insert([{**key, **cal} for cal in calibrations])

        # insert the recordings into the Recordings table
        self.Recordings.insert([{**key, **rec} for rec in recordings])