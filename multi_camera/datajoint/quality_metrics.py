import datajoint as dj
import numpy as np

from multi_camera.datajoint.multi_camera_dj import (
    PersonKeypointReconstruction,
    PersonKeypointReconstructionMethod,
    PersonKeypointReconstructionMethodLookup,
    SingleCameraVideo,
    MultiCameraRecording,
)
from pose_pipeline import TopDownPerson, TopDownMethodLookup, PersonBbox

schema = dj.schema("multicamera_tracking")


def interpolate_nan(x):
    # Create a mask for invalid measurements (True where values are NaN)
    mask = np.isnan(x)

    # Indices of the valid measurements
    valid_indices = np.where(~mask)[0]

    # Indices of the invalid measurements
    invalid_indices = np.where(mask)[0]

    # Perform the linear interpolation
    interpolated_values = np.interp(invalid_indices, valid_indices, x[~mask])

    # Fill in the interpolated values into the original time series
    x[mask] = interpolated_values
    return x


@schema
class SynchronizationQuality(dj.Computed):
    definition = """
    # Score the synchronization quality of the cameras
    -> MultiCameraRecording
    ---
    num_zero_timestamps     : int      # number of timestamps that were zero
    max_spread              : float    # max spread in values (in ms)
    mean_spread             : float    # mean spread in values (in ms)
    rms_spread              : float    # root mean square spread in values (in ms)
    fps                     : float    # estimated frames per second
    """

    def make(self, key):

        frame_timestamps = (SingleCameraVideo * MultiCameraRecording & key).fetch("frame_timestamps")
        frame_timestamps = np.stack(frame_timestamps)

        assert frame_timestamps.shape[0] > 1, "Failed to find multiple camera timestamps"

        number_of_zeros = np.sum(frame_timestamps == 0)

        # now replace all zeros with nan and interpolate them in
        invalid = frame_timestamps == 0
        frame_timestamps = frame_timestamps.astype(float)
        frame_timestamps[invalid] = np.nan

        for i in range(frame_timestamps.shape[0]):
            frame_timestamps[i] = interpolate_nan(frame_timestamps[i])

        frame_timestamps = frame_timestamps / 1e9  # convert from ns to seconds
        frame_timestamps = frame_timestamps * 1000.0  # convert to ms
        spread = np.max(frame_timestamps, axis=0) - np.min(frame_timestamps, axis=0)

        # TODO: should add some statistics around this, as it captures whether just a single
        # camera is having issues or all of them
        delta = frame_timestamps - np.mean(frame_timestamps, axis=0)

        fps = 1000 / np.mean(np.diff(frame_timestamps, axis=1))

        max_spread = np.max(spread)
        mean_spread = np.mean(spread)
        rms_spread = np.sqrt(np.mean(spread**2))

        key["num_zero_timestamps"] = number_of_zeros
        key["max_spread"] = max_spread
        key["mean_spread"] = mean_spread
        key["rms_spread"] = rms_spread
        key["fps"] = fps

        self.insert1(key)


# NOTE: we don't have a good grouping for detection for the multicamera system
# and so we follow the design pattern of the PersonKeypointReconstruction that
# uses the method table to specify the tracking method


@schema
class MultiTrackingDetection(dj.Computed):
    definition = """
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> PersonKeypointReconstructionMethod
    ---
    min_cameras         : int
    max_cameras         : int
    mean_cameras        : float
    missing_frames      : int
    detected_fraction   : float
    broken_frames       : int
    num_breaks          : int
    """

    def make(self, key):
        from pose_pipeline import PersonBbox

        present = (PersonBbox * SingleCameraVideo & key).fetch("present")
        present = np.stack(present)  # if this throws an error there are different lengths for different trials
        cameras = np.sum(present, axis=0)

        detected_thresold = 4
        min_cameras = np.min(cameras)
        max_cameras = np.max(cameras)
        mean_cameras = np.mean(cameras)

        detected = cameras >= detected_thresold
        missing_frames = np.sum(~detected)
        detected_fraction = np.mean(detected)

        # determine if there are any periods breaking the detection in the middle
        # first remove any continuous block of not-detected at the beginning
        if detected[0] == 0:
            breaks = np.diff(detected)
            start = np.argmax(breaks) + 1
            detected = detected[start:]
        if detected[-1] == 0:
            breaks = np.diff(detected)
            end = len(detected) - np.argmax(breaks[::-1]) - 1
            detected = detected[:end]
        breaks = np.diff(detected)

        broken_frames = np.sum(~detected)
        num_breaks = np.sum(breaks)

        key["min_cameras"] = min_cameras
        key["max_cameras"] = max_cameras
        key["mean_cameras"] = mean_cameras
        key["missing_frames"] = missing_frames
        key["detected_fraction"] = detected_fraction
        key["broken_frames"] = broken_frames
        key["num_breaks"] = num_breaks

        self.insert1(key)

    @property
    def key_source(self):
        # awkward double negative is to ensure all BlurredVideo views were computed
        return PersonKeypointReconstructionMethod - (PersonKeypointReconstructionMethod * SingleCameraVideo - TopDownPerson).proj()


def define_plane(points):
    """
    Defines a plane using an array of 3D points.

    Parameters:
        points (numpy.ndarray): An N x 3 matrix of coordinates.

    Returns:
        tuple:
            origin (numpy.ndarray): The centroid of the points.
            vector1 (numpy.ndarray): The first principal component.
            vector2 (numpy.ndarray): The second principal component.
    """

    from numpy.linalg import svd

    # Compute the centroid
    origin = np.mean(points, axis=0)

    # Center the points
    points_centered = points - origin

    # Compute SVD
    U, _, _ = svd(points_centered.T, full_matrices=False)

    # Extract the first two principal components, then define coordinate system
    x = U[:, 0]
    y = U[:, 1]
    z = np.cross(x, y)

    floor_orientation = np.stack([x, y, z])

    return origin, floor_orientation


@schema
class FloorLevel(dj.Computed):
    definition = """
    # Score how level the plane defined by the heels is, for crudely calibration
    -> PersonKeypointReconstructionMethod
    ---
    floor_angle       : float
    """

    def make(self, key):

        # NOTE: this assumes the person moves around and their feet are mostly on the
        # ground. so tasks like postural sway will perform poorly on this, even if
        # calibration is fine.

        joints = TopDownPerson.joint_names("Bridging_bml_movi_87")
        idx = np.array([joints.index(j) for j in ["Left Heel", "Right Heel"]])

        kp3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")

        kp3d = kp3d[:, idx]
        conf = np.min(kp3d[:, :, 3], axis=1) > 0.5
        kp3d = kp3d[conf, :, :3]

        # find the lowest point for each frame (presumes up is broadly correct)
        lowest = np.argmin(kp3d[:, :, 2], axis=1)
        lowest_kp3d = kp3d[np.arange(len(lowest)), lowest]

        # estimate the floor place and origin
        origin, floor_plane = define_plane(lowest_kp3d)

        # work out degrees from vertical
        z = floor_plane[2]
        z = z / np.linalg.norm(z) * np.sign(z[2])
        ang = np.arccos(np.dot(z, np.array([0, 0, 1])))
        ang = np.rad2deg(ang)

        key["floor_angle"] = ang
        self.insert1(key)

    @property
    def key_source(self):
        # forcing this to use our most common implicit method, as this produces more stable results
        method = (
            PersonKeypointReconstructionMethodLookup * TopDownMethodLookup
            & 'reconstruction_method_name="Implicit Optimization KP Conf, MaxHuber=10" and top_down_method_name="Bridging_bml_movi_87"'
        )
        return PersonKeypointReconstruction & method
