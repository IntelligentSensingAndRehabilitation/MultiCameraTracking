import datajoint as dj
import numpy as np

from .calibrate_cameras import Calibration
from pose_pipeline import Video, VideoInfo, TopDownPerson, TopDownMethodLookup, BestDetectedFrames

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

        import numpy as np
        from ..analysis.camera import triangulate_point

        calibration_key = (Calibration & key).fetch1('KEY')
        recording_key = (MultiCameraRecording & key).fetch1('KEY')
        top_down_method = key['top_down_method']

        camera_calibration, camera_names = (Calibration & calibration_key).fetch1('camera_calibration', 'camera_names')
        keypoints, camera_name = (TopDownPerson * SingleCameraVideo * MultiCameraRecording &
                                recording_key & {'top_down_method': top_down_method}).fetch('keypoints', 'camera_name')

        # if one video has one less frames then crop it
        N = min([k.shape[0] for k in keypoints])
        assert all([k.shape[0] - N < 2 for k in keypoints])
        keypoints = [k[:N] for k in keypoints]

        # work out the order that matches the calibration (should normally match)
        order = [list(camera_name).index(c) for c in camera_names]
        points2d = np.stack([keypoints[o][:, :, :] for o in order], axis=0)

        points3d = triangulate_point(camera_calibration, points2d)
        key['keypoints3d'] = np.array(points3d)

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
    vertices            : longblob
    faces               : longblob
    '''

    def make(self, key):
        from ..analysis.easymocap import easymocap_fit_smpl_3d, get_joint_openpose, get_vertices, get_faces
        from easymocap.dataset import CONFIG as config

        # get triangulated points and convert to meters
        points3d = (PersonKeypointReconstruction & key).fetch1('keypoints3d') / 1000.0

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
        key['vertices'] = get_vertices(res)
        key['faces'] = get_faces()
        self.insert1(key)

    def get_result(self):
        poses, shapes, Rh, Th = self.fetch1('poses', 'shape', 'orientation', 'translation')
        return {'poses': poses, 'shapes': shapes, 'Rh': Rh, 'Th': Th}

    def export_trc(self, filename, z_offset=0):
        from pose_pipeline import TopDownPerson, VideoInfo
        from multi_camera.analysis.opensim import normalize_marker_names, points3d_to_trc

        joint_names = TopDownPerson.joint_names('OpenPose')
        joints3d = self.fetch1('joints3d')
        fps = np.unique((VideoInfo * SingleCameraVideo * MultiCameraRecording & self).fetch('fps'))

        points3d_to_trc(joints3d + np.array([[[0, z_offset, 0]]]), filename,
                        normalize_marker_names(joint_names), fps=fps)


@schema
class SMPLReconstructionVideos(dj.Computed):
    definition = '''
    # Videos of SMPL reconstruction from multiview
    -> SMPLReconstruction
    ---
    '''

    class Video(dj.Part):
        definition = '''
        -> SMPLReconstructionVideos
        -> SingleCameraVideo
        ---
        output_video      : attach@localattach    # datajoint managed video file
        '''

    def make(self, key):
        import cv2
        import os
        import tempfile
        from einops import rearrange
        from ..analysis.camera import project_distortion, get_intrinsic, get_extrinsic, distort_3d
        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints
        from easymocap.visualize.renderer import Renderer

        self.insert1(key)

        videos = Video * TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & BestDetectedFrames & key
        video_keys, camera_names, keypoints2d = videos.fetch('KEY', 'camera_name', 'keypoints')
        camera_params = (Calibration & key).fetch1('camera_calibration')

        # get video parameters
        width = np.unique((VideoInfo & video_keys).fetch('width'))[0]
        height = np.unique((VideoInfo & video_keys).fetch('height'))[0]
        fps = np.unique((VideoInfo & video_keys).fetch('fps'))[0]

        # get vertices, in world coordintes
        faces, vertices, joints3d = (SMPLReconstruction & key).fetch1('faces', 'vertices', 'joints3d')

        # convert from meter to the mm that the camera model expects
        joints3d = joints3d * 1000.0

        # compute keypoints from reprojection of SMPL fit
        keypoints2d = np.array([project_distortion(camera_params, i, joints3d) for i in range(camera_params['mtx'].shape[0])])

        render = Renderer(height=height, width=width, down_scale=2, bg_color=[0, 0, 0, 0.0])

        for i, video_key in enumerate(video_keys[:1]):

            # get camera parameters
            K = np.array(get_intrinsic(camera_params, i))

            # don't use real extrinsic since we apply distortion which does this
            R = np.eye(3)
            T = np.zeros((3,))
            cameras = {'K': [K], 'R': [R], 'T': [T]}

            # account for camera distortion. convert vertices to mm first.
            vertices_distorted = np.array(distort_3d(camera_params, i, vertices * 1000.0))
            # then back to meters
            vertices_distorted = vertices_distorted / 1000.0

            def render_overlay(frame, idx, vertices=vertices_distorted, faces=faces, cameras=cameras):

                if idx >= vertices.shape[0]:
                    return frame

                render_data = {3: {'vertices': vertices[idx], 'faces': faces, 'name': 'human'}}

                frame = render.render(render_data, cameras, [frame], add_back=True)[0].copy()
                frame = draw_keypoints(frame, keypoints2d[i][idx] / render.down_scale, radius=6, color=(125, 125, 255))

                return frame

            fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            video = (Video & video_key).fetch1('video')
            video_overlay(video, out_file_name, render_overlay, max_frames=None, downsample=2, compress=True)

            single_video_key = (SingleCameraVideo * SMPLReconstruction & key & video_key).fetch1('KEY')
            single_video_key['output_video'] = out_file_name

            SMPLReconstructionVideos.Video.insert1(single_video_key)

            os.remove(video)
            os.remove(out_file_name)



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
