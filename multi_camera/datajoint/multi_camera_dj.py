import datajoint as dj
import numpy as np

from .calibrate_cameras import Calibration
from pose_pipeline import VideoInfo, TopDownPerson, TopDownMethodLookup

schema = dj.schema("multicamera_tracking")


@schema
class MultiCameraRecording(dj.Manual):
    definition = '''
    # Recording from multiple synchronized cameras
    recording_timestamp : timestamp
    camera_config_hash  : varchar(50)    # camera configuration
    video_project       : varchar(50)    # video project, which should match pose pipeline
    ---
    video_base_filename : varchar(100)   # base name for the videos without serial prefix
    '''


@schema
class SingleCameraVideo(dj.Manual):
    definition = """
    # Single view of a multiview recording
    -> MultiCameraRecording
    -> Video
    ---
    camera_name          : varchar(50)
    frame_timstamps      : longblob   # precise timestamps from that camera
    """


@schema
class CalibratedRecording(dj.Manual):
    definition = '''
    # Match calibration to a recording
    -> MultiCameraRecording
    -> Calibration
    '''


@schema
class PersonKeypointReconstructionMethod(dj.Manual):
    definition = '''
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> CalibratedRecording
    top_down_method     :  int
    '''


@schema
class PersonKeypointReconstruction(dj.Computed):
    definition = '''
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> PersonKeypointReconstructionMethod
    ---
    keypoints3d         : longblob
    '''

    def make(self, key):

        from .triangulate import reconstruct

        calibration_key = (Calibration & key).fetch1('KEY')
        recording_key = (MultiCameraRecording & key).fetch1('KEY')

        key['keypoints3d'] = reconstruct(recording_key, calibration_key, top_down_method=key['top_down_method'])
        self.insert1(key, allow_direct_insert=True)


@schema
class PersonKeypointReconstructionVideo(dj.Computed):
    definition = '''
    # Video from reconstruction
    -> PersonKeypointReconstruction
    ---
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):
        import os
        import tempfile
        import numpy as np
        from ..utils.visualization import skeleton_video

        method_name = (TopDownMethodLookup & key).fetch1('top_down_method_name')
        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        fps = np.unique((VideoInfo * SingleCameraVideo & key).fetch('fps'))[0]
        #fps = np.round(fps)[0]

        keypoints3d = (PersonKeypointReconstruction & key).fetch1('keypoints3d')
        skeleton_video(keypoints3d, out_file_name, method_name, fps=fps)

        key["output_video"] = out_file_name
        self.insert1(key)


@schema
class SMPLReconstruction(dj.Computed):
    definition = '''
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> PersonKeypointReconstruction
    ---
    poses               : longblob
    shape               : longblob
    orientation         : longblob
    translation         : longblob
    joints3d            : longblob
    '''

    def make(self, key):
        from ..analysis.easymocap import easymocap_fit_smpl_3d, get_joint_openpose
        from easymocap.dataset import CONFIG as config

        # get triangulated points and convert to meters
        points3d = (PersonKeypointReconstruction & key).fetch1('keypoints3d') / 100.0

        if key['top_down_method'] == 2:
            # Convert the HALPE coordinate order to the expected order

            def joint_renamer(j):
                j = j.replace('Sternum', 'Neck')
                j = j.replace('Right ', 'R')
                j = j.replace('Left ', 'L')
                j = j.replace('Little', 'Small')
                j = j.replace('Pelvis', 'MidHip')
                j = j.replace(' ', '')
                return j

            def normalize_marker_names(joints):
                """ Convert joint names to those expected by OpenSim model """
                return [joint_renamer(j) for j in joints]

            joint_names = normalize_marker_names(TopDownPerson.joint_names('MMPoseHalpe'))
            joint_reorder = np.array([joint_names.index(j) for j in config['body15']['joint_names']])
            points3d = points3d[:, joint_reorder]

        elif key['top_down_method'] == 4:
            # for OpenPose the keypoint order can be preserved
            pass

        else:
            raise NotImplementedError(f'Top down method {key["top_down_method"]} not supported.')

        res = easymocap_fit_smpl_3d(points3d, verbose=True)
        key['poses'] = res['poses']
        key['shape'] = res['shapes']
        key['orientation'] = res['Rh']
        key['translation'] = res['Th']
        key['joints3d'] = get_joint_openpose(res)
        self.insert1(key)

    def export_trc(self, filename, z_offset=0):
        from pose_pipeline import TopDownPerson, VideoInfo
        from multi_camera.analysis.opensim import normalize_marker_names, points3d_to_trc

        joint_names = TopDownPerson.joint_names('OpenPose')
        joints3d = self.fetch1('joints3d')
        fps = np.unique((VideoInfo * SingleCameraVideo * MultiCameraRecording & self).fetch('fps'))


        points3d_to_trc(joints3d + np.array([[[0, z_offset, 0]]]), filename,
                        normalize_marker_names(joint_names), fps=fps)


def import_recording(vid_base, vid_path='.', video_project='MULTICAMERA_TEST', legacy_flip=None):
    import os
    import json
    import numpy as np
    import datajoint as dj
    from datetime import datetime

    from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo
    from ..analysis.calibration import hash_names
    from pose_pipeline import Video

    # search for files. expects them to be in the format vid_base.serial_number.mp4
    vids = []
    camera_names = []
    for v in os.listdir(vid_path):
        base, ext = os.path.splitext(v)
        if ext == '.mp4' and len(base.split('.')) == 2 and base.split('.')[0] == vid_base:
            vids.append(os.path.join(vid_path, v))

    print(f'Found {len(vids)} videos.')

    def mysplit(x):
        splits = x.split('_')
        base = '_'.join(splits[:-2])
        date = '_'.join(splits[-2:])

        return base, date

    camera_names = [os.path.split(v)[1].split('.')[1] for v in vids]
    camera_hash = hash_names(camera_names)
    _, timestamp = mysplit(vid_base)
    timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')

    parent = {'recording_timestamps': timestamp, 'camera_config_hash': camera_hash, 'video_project': video_project,
              'video_base_filename': vid_base}

    timestamps = json.load(open(os.path.join(vid_path, vid_base + '.json'), 'r'))
    frame_timestamps = np.array(timestamps['timestamps'])
    if 'serials' in timestamps.keys():
        serials = timestamps['serials']
    else:
        assert legacy_flip is not None, "Please specify flip direction for videos without serial numbers"
        if legacy_flip:
            serials = ['UnknownLeft', 'UnknownRight']
        else:
            serials = ['UnknownRight', 'UnknownLeft']

    assert all(np.sort(serials) == np.sort(camera_names))

    vid_structs = []
    single_structs = []
    for v, serial in zip(vids, camera_names):

        vid_filename = os.path.split(v)[1]
        vid_filename = os.path.splitext(vid_filename)[0]

        vid_struct = {'video_project': video_project, 'filename': vid_filename,
                      'start_time': timestamp, 'video': v}

        ts_idx = serials.index(serial)
        single_struct = {'recording_timestamps': timestamp, 'camera_config_hash': camera_hash, 'camera_name': serial,
                         'video_project': video_project, 'filename': vid_filename, 'frame_timestamps': list(frame_timestamps[:, ts_idx])}

        vid_structs.append(vid_struct)
        single_structs.append(single_struct)

    dj.conn().start_transaction()
    try:
        MultiCameraRecording.insert1(parent)
        Video.insert(vid_structs, skip_duplicates=True)
        SingleCameraVideo.insert(single_structs)
    except Exception as e:
        dj.conn().cancel_transaction()
        raise e
    else:
        dj.conn().commit_transaction()
