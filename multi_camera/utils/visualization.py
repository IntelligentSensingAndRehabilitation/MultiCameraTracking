import os
import cv2
import numpy as np
import concurrent
import tempfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from pose_pipeline import OpenPosePerson, TopDownPerson, LiftingPerson


def center_skeleton(keypoints3d, joints):
    joints = [j.upper() for j in joints]
    pelvis = np.mean(
        keypoints3d[:, np.array([joints.index("LEFT HIP"), joints.index("RIGHT HIP")])], axis=1, keepdims=True
    )
    centered = keypoints3d - pelvis

    return centered


def skeleton_video(keypoints3d, filename, method, fps=30.0):

    if method == "OpenPose":
        joints = OpenPosePerson.joint_names()
        left = [
            "Left Little Toe",
            "Left Ankle",
            "Left Big Toe",
            "Left Ankle",
            "Left Heel",
            "Left Ankle",
            "Left Knee",
            "Left Hip",
            "Pelvis",
            "Sternum",
            "Left Shoulder",
            "Left Elbow",
            "Left Wrist",
        ]
        right = [
            "Right Little Toe",
            "Right Ankle",
            "Right Big Toe",
            "Right Ankle",
            "Right Heel",
            "Right Ankle",
            "Right Knee",
            "Right Hip",
            "Pelvis",
            "Sternum",
            "Right Shoulder",
            "Right Elbow",
            "Right Wrist",
        ]
    elif method == "MMPose":
        joints = TopDownPerson.joint_names("MMPoseCoco")
        left = ["Left Ankle", "Left Knee", "Left Hip", "Left Shoulder", "Left Elbow", "Left Wrist"]
        right = ["Right Ankle", "Right Knee", "Right Hip", "Right Shoulder", "Right Elbow", "Right Wrist"]
    elif method == "MMPoseWholebody":
        joints = TopDownPerson.joint_names("MMPoseWholebody")
        left = [
            "Left Little Toe",
            "Left Ankle",
            "Left Big Toe",
            "Left Ankle",
            "Left Heel",
            "Left Ankle",
            "Left Knee",
            "Left Hip",
            "Pelvis",
            "Sternum",
            "Left Shoulder",
            "Left Elbow",
            "Left Wrist",
        ]
        right = [
            "Right Little Toe",
            "Right Ankle",
            "Right Big Toe",
            "Right Ankle",
            "Right Heel",
            "Right Ankle",
            "Right Knee",
            "Right Hip",
            "Pelvis",
            "Sternum",
            "Right Shoulder",
            "Right Elbow",
            "Right Wrist",
        ]
    elif method == "MMPoseHalpe":
        joints = TopDownPerson.joint_names("MMPoseHalpe")
        left = [
            "Left Ankle",
            "Left Little Toe",
            "Left Big Toe",
            "Left Ankle",
            "Left Heel",
            "Left Ankle",
            "Left Knee",
            "Left Hip",
            "Pelvis",
            "Neck",
            "Head",
            "Neck",
            "Left Shoulder",
            "Left Elbow",
            "Left Wrist",
        ]
        right = [
            "Right Ankle",
            "Right Little Toe",
            "Right Big Toe",
            "Right Ankle",
            "Right Heel",
            "Right Ankle",
            "Right Knee",
            "Right Hip",
            "Pelvis",
            "Neck",
            "Head",
            "Neck",
            "Right Shoulder",
            "Right Elbow",
            "Right Wrist",
        ]
    elif method == "OpenPose_BODY25B":
        joints = TopDownPerson.joint_names("OpenPose_BODY25B")
        left = [
            "Left Ankle",
            "Left Little Toe",
            "Left Big Toe",
            "Left Ankle",
            "Left Heel",
            "Left Ankle",
            "Left Knee",
            "Left Hip",
            "Neck",
            "Head",
            "Neck",
            "Left Shoulder",
            "Left Elbow",
            "Left Wrist",
        ]
        right = [
            "Right Ankle",
            "Right Little Toe",
            "Right Big Toe",
            "Right Ankle",
            "Right Heel",
            "Right Ankle",
            "Right Knee",
            "Right Hip",
            "Neck",
            "Head",
            "Neck",
            "Right Shoulder",
            "Right Elbow",
            "Right Wrist",
        ]
    elif method == "GastNet":
        joints = LiftingPerson.joint_names()
        left = [
            "Left foot",
            "Left knee",
            "Left hip",
            "Hip (root)",
            "Spine",
            "Thorax",
            "Head",
            "Nose",
            "Head",
            "Thorax",
            "Left shoulder",
            "Left elbow",
            "Left wrist",
        ]
        right = [
            "Right foot",
            "Right knee",
            "Right hip",
            "Hip (root)",
            "Spine",
            "Thorax",
            "Head",
            "Nose",
            "Head",
            "Thorax",
            "Right shoulder",
            "Right elbow",
            "Right wrist",
        ]
    else:
        raise Exception(f"Unknown method: {method}")

    centered = center_skeleton(keypoints3d, joints)

    left = np.array([joints.index(l) for l in left])
    right = np.array([joints.index(l) for l in right])
    num_main_joints = len(joints)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    def initialize():
        frame_idx = keypoints3d.shape[0] // 2
        lines = []

        lines.append(
            plt.plot(
                centered[frame_idx, :num_main_joints, 0],
                centered[frame_idx, :num_main_joints, 1],
                centered[frame_idx, :num_main_joints, 2],
                ".",
            )
        )
        lines.append(
            plt.plot(centered[frame_idx, left, 0], centered[frame_idx, left, 1], centered[frame_idx, left, 2], "b")
        )
        lines.append(
            plt.plot(centered[frame_idx, right, 0], centered[frame_idx, right, 1], centered[frame_idx, right, 2], "r")
        )
        lines.append(
            plt.plot(
                centered[frame_idx, num_main_joints:, 0],
                centered[frame_idx, num_main_joints:, 1],
                centered[frame_idx, num_main_joints:, 2],
                ".",
                markersize=1,
            )
        )

        return lines

    def update_video(frame_idx, lines):
        lines = lines.copy()

        lines[0][0].set_xdata(centered[frame_idx, :num_main_joints, 0])
        lines[0][0].set_ydata(centered[frame_idx, :num_main_joints, 1])
        lines[0][0].set_3d_properties(centered[frame_idx, :num_main_joints, 2], zdir="z")

        lines[1][0].set_xdata(centered[frame_idx, left, 0])
        lines[1][0].set_ydata(centered[frame_idx, left, 1])
        lines[1][0].set_3d_properties(centered[frame_idx, left, 2], zdir="z")

        lines[2][0].set_xdata(centered[frame_idx, right, 0])
        lines[2][0].set_ydata(centered[frame_idx, right, 1])
        lines[2][0].set_3d_properties(centered[frame_idx, right, 2], zdir="z")

        lines[3][0].set_xdata(centered[frame_idx, num_main_joints:, 0])
        lines[3][0].set_ydata(centered[frame_idx, num_main_joints:, 1])
        lines[3][0].set_3d_properties(centered[frame_idx, num_main_joints:, 2], zdir="z")

        # ax.view_init(elev=10., azim=-frame_idx/30*np.pi*2)
        ax.view_init(elev=10.0, azim=0)  # -frame_idx/30*np.pi*2)

    lines = initialize()

    def set_axes_equal(ax):
        """Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = 0  # np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = 0  # np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = 0  # np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        plot_radius = min([plot_radius, 1000])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    set_axes_equal(ax)

    from matplotlib.animation import FuncAnimation, writers

    anim = FuncAnimation(
        fig, update_video, frames=np.arange(0, keypoints3d.shape[0]), interval=1000 / fps, repeat=False, fargs=[lines]
    )

    Writer = writers["ffmpeg"]
    writer = Writer(fps=fps, metadata={}, bitrate=500000)
    anim.save(filename, writer=writer)

    return anim


def make_reprojection_video(
    key: dict,
    portrait_width=288,
    dilate=1.1,
    return_results=False,
    detected_keypoint_size=4,
    projected_keypoint_size=6,
    visible_threshold=0.35,
):
    """Create a video showing the cropped individual with the 2D keypoints project from each view"""

    from pose_pipeline import PersonBbox, BlurredVideo, TopDownPerson, Video, VideoInfo
    from pose_pipeline.utils.bounding_box import crop_image_bbox
    from multi_camera.datajoint.multi_camera_dj import (
        MultiCameraRecording,
        PersonKeypointReconstruction,
        SingleCameraVideo,
        Calibration,
    )
    from ..analysis.camera import project_distortion
    from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

    recording_fn = (MultiCameraRecording & key).fetch1("video_base_filename")
    videos = TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key
    video_keys, video_camera_name = (TopDownPerson.proj() * SingleCameraVideo.proj() * videos).fetch(
        "KEY", "camera_name"
    )
    keypoints3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")
    camera_params, camera_names = (Calibration & key).fetch1("camera_calibration", "camera_names")
    assert camera_names == video_camera_name.tolist(), "Videos don't match cameras in calibration"

    # get video parameters
    width = np.unique((VideoInfo & video_keys).fetch("width"))[0]
    height = np.unique((VideoInfo & video_keys).fetch("height"))[0]
    fps = np.unique((VideoInfo & video_keys).fetch("fps"))[0]

    # compute keypoints from reprojection of SMPL fit
    kp3d = keypoints3d[..., :-1]
    conf3d = keypoints3d[..., -1]
    keypoints2d = np.array([project_distortion(camera_params, i, kp3d) for i in range(camera_params["mtx"].shape[0])])

    # handle any bad projections
    valid_kp = np.tile((conf3d < 0.5)[None, ...], [keypoints2d.shape[0], 1, 1])
    clipped = np.logical_or.reduce(
        (
            keypoints2d[..., 0] <= 0,
            keypoints2d[..., 0] >= width,
            keypoints2d[..., 1] <= 0,
            keypoints2d[..., 1] >= height,
            np.isnan(keypoints2d[..., 0]),
            np.isnan(keypoints2d[..., 1]),
            valid_kp,
        )
    )
    keypoints2d[clipped, 0] = 0
    keypoints2d[clipped, 1] = 0
    # add low confidence when clipped
    keypoints2d = np.concatenate([keypoints2d, ~clipped[..., None] * 1.0], axis=-1)

    total_frames = min(kp3d.shape[0], np.min((VideoInfo & video_keys).fetch("num_frames")))

    bbox_fns = [PersonBbox.get_overlay_fn(v) for v in video_keys]
    videos = [(BlurredVideo & v).fetch1("output_video") for v in video_keys]
    bboxes = [(PersonBbox & v).fetch1("bbox").astype(int) for v in video_keys]
    kp2d_detected = np.array([(TopDownPerson & v).fetch1("keypoints")[:total_frames] for v in video_keys])

    def make_frames(video_idx):

        video = videos[video_idx]
        cap = cv2.VideoCapture(video)
        bbox_fn = bbox_fns[video_idx]
        bbox = bboxes[video_idx]

        results = []

        if video_idx == 0:
            iter = tqdm(range(total_frames), desc=f"Extracting frames {recording_fn}")
        else:
            iter = range(total_frames)

        for frame_idx in iter:
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = draw_keypoints(
                frame,
                keypoints2d[video_idx, frame_idx],
                radius=projected_keypoint_size,
                color=(125, 125, 255),
                threshold=visible_threshold,
            )
            frame = draw_keypoints(
                frame,
                kp2d_detected[video_idx, frame_idx],
                radius=detected_keypoint_size,
                color=(255, 80, 80),
                border_color=(64, 20, 20),
                threshold=visible_threshold,
            )
            frame = bbox_fn(frame, frame_idx, width=2, color=(0, 0, 255))
            frame = crop_image_bbox(
                frame, bbox[frame_idx], target_size=(portrait_width, int(portrait_width * 1920 / 1080)), dilate=dilate
            )[0]

            results.append(frame)

        cap.release()
        os.remove(video)

        return results

    # use multithreading futures to run make_frames for each video in parallel and collate teh results
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(make_frames, range(len(video_keys))))

    def images_to_grid(images, n_cols=5):
        n_rows = int(np.ceil(len(images) / n_cols))
        grid = np.zeros((n_rows * images[0].shape[0], n_cols * images[0].shape[1], 3), dtype=np.uint8)
        for i, img in enumerate(images):
            row = i // n_cols
            col = i % n_cols
            grid[row * img.shape[0] : (row + 1) * img.shape[0], col * img.shape[1] : (col + 1) * img.shape[1], :] = img
        return grid

    # collate the results into a grid
    results = [images_to_grid(r) for r in zip(*results)]

    # write the collated frames into a video matching the original frame rate using opencv VideoWriter
    fd, filename = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, fps, (results[0].shape[1], results[0].shape[0]))
    for frame in tqdm(results, desc="Writing"):
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    from pose_pipeline.utils.video_format import compress

    compressed_filename = compress(filename)
    os.remove(filename)

    if return_results:
        return compressed_filename, results
    return compressed_filename
