
import pose_pipeline
from pose_pipeline.pipeline import Video, VideoInfo
from pose_pipeline.utils.standard_pipelines import bottom_up_pipeline
from multi_camera.datajoint.sessions import Subject, Recording
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo, CalibratedRecording, Calibration
from multi_camera.datajoint.easymocap import EasymocapTracking, EasymocapSmpl
pose_pipeline.set_environmental_variables()


def assign_calibration():
    # find the calibration that is closest in time to each recording that also has a minimum
    # threshold. note that this will possibly allow different calibrations within a session,
    # but this is intentional to also handle using different camera setups in a session

    missing = (MultiCameraRecording & Recording - CalibratedRecording).proj()
    calibration_offset = (Calibration * MultiCameraRecording).proj(calibration_offset = 'ABS(recording_timestamps-cal_timestamp)')

    # only accept low reprojection errors
    calibration_offset = calibration_offset & ( Calibration  & 'reprojection_error < 0.2' )

    # find closest calibrations to an experiment
    viable = missing.aggr(calibration_offset, min_calibration_offset='MIN(calibration_offset)') & 'min_calibration_offset < 20000'

    # need to awkwardly use the fetch at end to join on dependent attributes. this is basically performing an
    # argmin operation
    viable = viable.proj(calibration_offset='min_calibration_offset')
    matches = calibration_offset & viable.fetch(as_dict=True)

    CalibratedRecording.insert(matches.fetch("KEY"), skip_duplicates=True)


def preannotation_session_pipeline(bridging=True):
    keys = (Video & (SingleCameraVideo & Recording)).fetch('KEY')

    if bridging:
        bottom_up_pipeline(keys, bottom_up_method_name ="Bridging_OpenPose")
    else:
        bottom_up_pipeline(keys, bottom_up_method_name ="OpenPose_HR")

    # now run easymocap
    VideoInfo.populate(SingleCameraVideo * MultiCameraRecording, reserve_jobs=True)
    EasymocapTracking.populate(MultiCameraRecording * CalibratedRecording, reserve_jobs=True, suppress_errors=True)
    EasymocapSmpl.populate(MultiCameraRecording * CalibratedRecording, reserve_jobs=True, suppress_errors=True)


if __name__ == "__main__":
    assign_calibration()
    preannotation_session_pipeline()
