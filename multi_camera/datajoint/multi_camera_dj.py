import datajoint as dj
import numpy as np

from .calibrate_cameras import Calibration
from pose_pipeline import Video, VideoInfo, TopDownPerson, TopDownMethodLookup, BestDetectedFrames, BlurredVideo

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
    tracking_method     :  int
    top_down_method     :  int
    '''


@schema
class PersonKeypointReconstruction(dj.Computed):
    definition = '''
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> PersonKeypointReconstructionMethod
    ---
    keypoints3d         : longblob
    camera_weights      : longblob
    '''

    def make(self, key):

        import numpy as np
        from ..analysis.camera import robust_triangulate_points

        calibration_key = (Calibration & key).fetch1('KEY')
        recording_key = (MultiCameraRecording & key).fetch1('KEY')
        top_down_method = key['top_down_method']
        tracking_method = key['tracking_method']

        camera_calibration, camera_names = (Calibration & calibration_key).fetch1('camera_calibration', 'camera_names')
        keypoints, camera_name = (TopDownPerson * SingleCameraVideo * MultiCameraRecording &
                                 {'top_down_method': top_down_method, 'tracking_method': tracking_method} &
                                 recording_key).fetch('keypoints', 'camera_name')

        # need to add zeros to missing frames since they occurr at the beginning of videos
        N = max([len(k) for k in keypoints])
        keypoints = np.stack([np.concatenate([np.zeros([N-k.shape[0], *k.shape[1:]]), k], axis=0) for k in keypoints], axis=0)

        print(len(camera_names), len(camera_name))
        # work out the order that matches the calibration (should normally match)
        order = [list(camera_name).index(c) for c in camera_names]
        points2d = np.stack([keypoints[o][:, :, :] for o in order], axis=0)

        points3d, camera_weights = robust_triangulate_points(camera_calibration, points2d, return_weights=True)
        key['keypoints3d'] = np.array(points3d)
        key['camera_weights'] = np.array(camera_weights)

        self.insert1(key, allow_direct_insert=True)

    def export_trc(self, filename, z_offset=0, start=None, end=None, return_points=False, smooth=False):
        ''' Export an OpenSim file of marker trajectories

            Params:
                filename (string) : filename to export to
                z_offset (float, optional) : optional vertical offset
                start    (float, optional) : if set, time to start at
                end      (float, optional) : if set, time to end at
                return_points (bool, opt)  : if true, return points
        '''

        from pose_pipeline import TopDownPerson, VideoInfo
        from multi_camera.analysis.opensim import normalize_marker_names, points3d_to_trc

        method_name = (TopDownMethodLookup & self).fetch1('top_down_method_name')
        joint_names = TopDownPerson.joint_names(method_name)

        joints3d = self.fetch1('keypoints3d').copy()
        joints3d = joints3d[:, :len(joint_names)] # discard "unnamed" joints
        joints3d = joints3d / 1000.0 # convert to m
        fps = np.unique((VideoInfo * SingleCameraVideo * MultiCameraRecording & self).fetch('fps'))

        if joints3d.shape[-1] == 4:
            joints3d = joints3d[..., :-1]

        if end is not None:
            joints3d = joints3d[:int(end * fps)]
        if start is not None:
            joints3d = joints3d[int(start*fps):]

        if smooth:
            import scipy

            for i in range(joints3d.shape[1]):
                for j in range(joints3d.shape[2]):
                    joints3d[:, i, j] = scipy.signal.medfilt(joints3d[:, i, j], 5)

        points3d_to_trc(joints3d + np.array([[[0, z_offset, 0]]]), filename,
                        normalize_marker_names(joint_names), fps=fps)

        if return_points:
            return joints3d

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
class PersonKeypointReprojectionVideos(dj.Computed):
    definition = '''
    # Videos of reconstruction preprojections
    -> PersonKeypointReconstruction
    ---
    '''

    class Video(dj.Part):
        definition = '''
        -> PersonKeypointReprojectionVideos
        -> SingleCameraVideo
        ---
        output_video      : attach@localattach    # datajoint managed video file
        '''

    def make(self, key):
        import cv2
        import os
        import tempfile
        from ..analysis.camera import project_distortion
        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

        self.insert1(key)

        videos = Video * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key
        video_keys, video_camera_name = (SingleCameraVideo.proj() * videos).fetch('KEY', 'camera_name')
        keypoints3d = (PersonKeypointReconstruction & key).fetch1('keypoints3d')
        camera_params, camera_names = (Calibration & key).fetch1('camera_calibration', 'camera_names')
        assert camera_names == video_camera_name.tolist(), "Videos don't match cameras in calibration"

        # get video parameters
        width = np.unique((VideoInfo & video_keys).fetch('width'))[0]
        height = np.unique((VideoInfo & video_keys).fetch('height'))[0]
        fps = np.unique((VideoInfo & video_keys).fetch('fps'))[0]

        # compute keypoints from reprojection of SMPL fit
        kp3d = keypoints3d[..., :-1]
        conf3d = keypoints3d[..., -1]
        keypoints2d = np.array([project_distortion(camera_params, i, kp3d) for i in range(camera_params['mtx'].shape[0])])

        print(f'Height: {height}. Width: {width}. FPS: {fps}')

        # handle any bad projections
        valid_kp = np.tile((conf3d < 0.5)[None, ...], [keypoints2d.shape[0], 1, 1])
        clipped = np.logical_or.reduce((keypoints2d[..., 0] <= 0, keypoints2d[..., 0] >= width,
                                       keypoints2d[..., 1] <= 0, keypoints2d[..., 1] >= height,
                                       np.isnan(keypoints2d[..., 0]), np.isnan(keypoints2d[..., 1]),
                                       valid_kp))
        keypoints2d[clipped, 0] = 0
        keypoints2d[clipped, 1] = 0
        # add low confidence when clipped
        keypoints2d = np.concatenate([keypoints2d, ~clipped[..., None] * 1.0], axis=-1)

        for i, video_key in enumerate(video_keys):

            def render_overlay(frame, idx):

                if idx >= keypoints2d.shape[1]:
                    return frame

                frame = draw_keypoints(frame, keypoints2d[i, idx], radius=6, color=(125, 125, 255), threshold=0.75)

                return frame

            fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            video = (BlurredVideo & video_key).fetch1('output_video')
            video_overlay(video, out_file_name, render_overlay, max_frames=None, downsample=2, compress=True)

            single_video_key = (SingleCameraVideo * PersonKeypointReconstruction & key & video_key).fetch1('KEY')
            single_video_key['output_video'] = out_file_name

            PersonKeypointReprojectionVideos.Video.insert1(single_video_key)

            os.remove(video)
            os.remove(out_file_name)


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
        points3d = (PersonKeypointReconstruction & key).fetch1('keypoints3d').copy()
        points3d[..., :3] = points3d[..., :3] / 1000.0  # convert coordinates to m, but leave confidence untouched

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

            # move these to where COCO would put them (which is what Body25 uses)
            points3d[:, joint_names.index('Neck')] = (points3d[:, joint_names.index('RShoulder')] + points3d[:, joint_names.index('LShoulder')]) / 2
            points3d[:, joint_names.index('MidHip')] = (points3d[:, joint_names.index('RHip')] + points3d[:, joint_names.index('LHip')]) / 2
            # reduce confidence on little toes as it seems to lock onto values from big toe (quick with MMPose model)
            points3d[:, joint_names.index('RSmallToe'), -1] = points3d[:, joint_names.index('RSmallToe'), -1] * 0.5
            points3d[:, joint_names.index('LSmallToe'), -1] = points3d[:, joint_names.index('LSmallToe'), -1] * 0.5

            joint_reorder = np.array([joint_names.index(j) for j in config['body25']['joint_names']])
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

    def export_trc(self, filename, z_offset=0, start=None, end=None, return_points=False):
        ''' Export an OpenSim file of marker trajectories

            Params:
                filename (string) : filename to export to
                z_offset (float, optional) : optional vertical offset
                start    (float, optional) : if set, time to start at
                end      (float, optional) : if set, time to end at
                return_points (bool, opt)  : if true, return points
        '''

        from pose_pipeline import TopDownPerson, VideoInfo
        from multi_camera.analysis.opensim import normalize_marker_names, points3d_to_trc

        joint_names = TopDownPerson.joint_names('OpenPose')
        joints3d = self.fetch1('joints3d')
        fps = np.unique((VideoInfo * SingleCameraVideo * MultiCameraRecording & self).fetch('fps'))

        if end is not None:
            joints3d = joints3d[:int(end * fps)]
        if start is not None:
            joints3d = joints3d[int(start*fps):]

        points3d_to_trc(joints3d + np.array([[[0, z_offset, 0]]]), filename,
                        normalize_marker_names(joint_names), fps=fps)

        if return_points:
            return joints3d


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

        videos = Video * TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key
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

        for i, video_key in enumerate(video_keys):

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
                frame = draw_keypoints(frame, keypoints2d[i][idx] / render.down_scale, radius=2, color=(125, 125, 255))

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


@schema
class SMPLXReconstruction(dj.Computed):
    definition = '''
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> PersonKeypointReconstruction
    ---
    poses               : longblob
    shape               : longblob
    orientation         : longblob
    translation         : longblob
    expression          : longblob
    joints3d            : longblob
    vertices            : longblob
    faces               : longblob
    '''

    def make(self, key):
        from ..analysis.easymocap import easymocap_fit_smpl_3d, get_joint_openpose, get_vertices, get_faces
        from easymocap.dataset import CONFIG as config

        # get triangulated points and convert to meters
        points3d = (PersonKeypointReconstruction & key).fetch1('keypoints3d').copy()
        points3d[..., :3] = points3d[..., :3] / 1000.0  # convert coordinates to m, but leave confidence untouched

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

            # move these to where COCO would put them (which is what Body25 uses)
            points3d[:, joint_names.index('Neck')] = (points3d[:, joint_names.index('RShoulder')] + points3d[:, joint_names.index('LShoulder')]) / 2
            points3d[:, joint_names.index('MidHip')] = (points3d[:, joint_names.index('RHip')] + points3d[:, joint_names.index('LHip')]) / 2
            # reduce confidence on little toes as it seems to lock onto values from big toe (quick with MMPose model)
            points3d[:, joint_names.index('RSmallToe'), -1] = points3d[:, joint_names.index('RSmallToe'), -1] * 0.1
            points3d[:, joint_names.index('LSmallToe'), -1] = points3d[:, joint_names.index('LSmallToe'), -1] * 0.1

            joint_reorder = np.array([joint_names.index(j) for j in config['body25']['joint_names']])
            points3d_body25 = points3d[:, joint_reorder]

            # from https://github.com/Fang-Haoshu/Halpe-FullBody
            left_hand = points3d[:, np.arange(94,115)]
            right_hand = points3d[:, np.arange(115,136)]
            # leave the first 17 points off. doesn't use outline. add 2 points at
            # end that halpe is missing
            face = points3d[:, np.arange(26+17, 94)]

            points3d = np.concatenate([points3d_body25, left_hand, right_hand, face], axis=1)

        elif key['top_down_method'] == 4:
            # for OpenPose the keypoint order can be preserved
            pass

        else:
            raise NotImplementedError(f'Top down method {key["top_down_method"]} not supported.')

        res = easymocap_fit_smpl_3d(points3d, verbose=True, body_model='smplx', skel_type='facebodyhand')
        key['poses'] = res['poses']
        key['shape'] = res['shapes']
        key['expression'] = res['expression']
        key['orientation'] = res['Rh']
        key['translation'] = res['Th']
        key['joints3d'] = get_joint_openpose(res, body_model='smplx')
        key['vertices'] = get_vertices(res, body_model='smplx')
        key['faces'] = get_faces(body_model='smplx')
        self.insert1(key)


@schema
class SMPLXReconstructionVideos(dj.Computed):
    definition = '''
    # Videos of SMPL reconstruction from multiview
    -> SMPLXReconstruction
    ---
    '''

    class Video(dj.Part):
        definition = '''
        -> SMPLXReconstructionVideos
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

        videos = Video * TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key
        video_keys, camera_names, keypoints2d = videos.fetch('KEY', 'camera_name', 'keypoints')
        camera_params = (Calibration & key).fetch1('camera_calibration')

        # get video parameters
        width = np.unique((VideoInfo & video_keys).fetch('width'))[0]
        height = np.unique((VideoInfo & video_keys).fetch('height'))[0]
        fps = np.unique((VideoInfo & video_keys).fetch('fps'))[0]

        # get vertices, in world coordintes
        faces, vertices, joints3d = (SMPLXReconstruction & key).fetch1('faces', 'vertices', 'joints3d')

        # convert from meter to the mm that the camera model expects
        joints3d = joints3d * 1000.0

        # compute keypoints from reprojection of SMPL fit
        keypoints2d = np.array([project_distortion(camera_params, i, joints3d) for i in range(camera_params['mtx'].shape[0])])

        render = Renderer(height=height, width=width, down_scale=2, bg_color=[0, 0, 0, 0.0])

        for i, video_key in enumerate(video_keys):

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
                #frame = draw_keypoints(frame, keypoints2d[i][idx] / render.down_scale, radius=1, color=(125, 125, 255))

                return frame

            fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            video = (Video & video_key).fetch1('video')
            video_overlay(video, out_file_name, render_overlay, max_frames=None, downsample=2, compress=True)

            single_video_key = (SingleCameraVideo * SMPLReconstruction & key & video_key).fetch1('KEY')
            single_video_key['output_video'] = out_file_name

            SMPLXReconstructionVideos.Video.insert1(single_video_key)

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
