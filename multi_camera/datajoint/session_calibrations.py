import datajoint as dj

from .calibrate_cameras import Calibration
from .multi_camera_dj import MultiCameraRecording, CalibratedRecording
from .sessions import schema, Session, Recording

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

        # Get a count of the number of calibrations associated with each recording
        session_recording = MultiCameraRecording * Recording & key
        calibrations_per_recording = Recording.aggr(CalibratedRecording & session_recording, n="count(*)")

        # See if there are any recordings with more or less than one calibration
        assert len(calibrations_per_recording & "n != 1") == 0, "Some recordings are associated with more than one calibration"
        assert len(calibrations_per_recording) == len(session_recording), "Some recordings are not associated with a calibration"

        # insert the key into the SessionCalibration table
        self.insert1(key)

        # Join the Recording and CalibratedRecording tables
        joined_calibrated_recordings = (Recording * CalibratedRecording & key).fetch("KEY")

        # get the calibrations from the joined_calibrated_recordings
        calibrations = (Calibration & joined_calibrated_recordings).fetch("KEY")

        # insert the calibrations into the Grouping table
        self.Grouping.insert([{**key, **cal} for cal in calibrations])

        # insert the recordings into the Recordings table
        self.Recordings.insert([{**key, **cal_rec} for cal_rec in joined_calibrated_recordings])