import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pose_pipeline import OpenPosePerson, TopDownPerson


def center_skeleton(keypoints3d, joints):
    pelvis = np.mean(keypoints3d[:, np.array([joints.index('Left Hip'), joints.index('Right Hip')])], axis=1, keepdims=True)
    centered = keypoints3d - pelvis

    return centered


def skeleton_video(keypoints3d, filename, method, fps=30.0):

    if method == 'OpenPose':
        joints = OpenPosePerson.joint_names()
        left = ['Left Little Toe', 'Left Ankle', 'Left Big Toe', 'Left Ankle', 'Left Heel', 'Left Ankle', 'Left Knee', 'Left Hip', 'Pelvis', 'Sternum', 'Left Shoulder', 'Left Elbow', 'Left Wrist']
        right = ['Right Little Toe', 'Right Ankle', 'Right Big Toe', 'Right Ankle', 'Right Heel', 'Right Ankle', 'Right Knee', 'Right Hip', 'Pelvis', 'Sternum', 'Right Shoulder', 'Right Elbow', 'Right Wrist']
    elif method == 'MMPose':
        joints = TopDownPerson.joint_names('MMPoseCoco')
        left = ['Left Ankle', 'Left Knee', 'Left Hip', 'Left Shoulder', 'Left Elbow', 'Left Wrist']
        right = [ 'Right Ankle', 'Right Knee', 'Right Hip', 'Right Shoulder', 'Right Elbow', 'Right Wrist']
    elif method == 'MMPoseWholebody':
        joints = TopDownPerson.joint_names('MMPoseWholebody')
        left = ['Left Little Toe', 'Left Ankle', 'Left Big Toe', 'Left Ankle', 'Left Heel', 'Left Ankle', 'Left Knee', 'Left Hip', 'Pelvis', 'Sternum', 'Left Shoulder', 'Left Elbow', 'Left Wrist']
        right = ['Right Little Toe', 'Right Ankle', 'Right Big Toe', 'Right Ankle', 'Right Heel', 'Right Ankle', 'Right Knee', 'Right Hip', 'Pelvis', 'Sternum', 'Right Shoulder', 'Right Elbow', 'Right Wrist']
    elif method == 'MMPoseHalpe':
        joints = TopDownPerson.joint_names('MMPoseHalpe')
        left = ['Left Ankle', 'Left Little Toe',  'Left Big Toe', 'Left Ankle', 'Left Heel', 'Left Ankle', 'Left Knee', 'Left Hip', 'Pelvis', 'Neck', 'Head', 'Neck', 'Left Shoulder', 'Left Elbow', 'Left Wrist']
        right = ['Right Ankle', 'Right Little Toe', 'Right Big Toe', 'Right Ankle', 'Right Heel', 'Right Ankle', 'Right Knee', 'Right Hip', 'Pelvis', 'Neck', 'Head', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist']
    else:
        raise Exception(f'Unknown method: {method}')

    centered = center_skeleton(keypoints3d, joints)

    left = np.array([joints.index(l) for l in left])
    right = np.array([joints.index(l) for l in right])
    num_main_joints = len(joints)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def initialize():
        frame_idx = keypoints3d.shape[0] // 2
        lines = []

        lines.append(plt.plot(centered[frame_idx, :num_main_joints, 0], centered[frame_idx, :num_main_joints, 1], centered[frame_idx, :num_main_joints, 2], '.'))
        lines.append(plt.plot(centered[frame_idx, left, 0], centered[frame_idx, left, 1], centered[frame_idx, left, 2], 'b'))
        lines.append(plt.plot(centered[frame_idx, right, 0], centered[frame_idx, right, 1], centered[frame_idx, right, 2], 'r'))
        lines.append(plt.plot(centered[frame_idx, num_main_joints:, 0], centered[frame_idx, num_main_joints:, 1], centered[frame_idx, num_main_joints:, 2], '.', markersize=1))

        return lines

    def update_video(frame_idx, lines):
        lines = lines.copy()

        lines[0][0].set_xdata(centered[frame_idx, :num_main_joints, 0])
        lines[0][0].set_ydata(centered[frame_idx, :num_main_joints, 1])
        lines[0][0].set_3d_properties(centered[frame_idx, :num_main_joints, 2], zdir='z')

        lines[1][0].set_xdata(centered[frame_idx, left, 0])
        lines[1][0].set_ydata(centered[frame_idx, left, 1])
        lines[1][0].set_3d_properties(centered[frame_idx, left, 2], zdir='z')

        lines[2][0].set_xdata(centered[frame_idx, right, 0])
        lines[2][0].set_ydata(centered[frame_idx, right, 1])
        lines[2][0].set_3d_properties(centered[frame_idx, right, 2], zdir='z')

        lines[3][0].set_xdata(centered[frame_idx, num_main_joints:, 0])
        lines[3][0].set_ydata(centered[frame_idx, num_main_joints:, 1])
        lines[3][0].set_3d_properties(centered[frame_idx, num_main_joints:, 2], zdir='z')


        ax.view_init(elev=10., azim=-frame_idx/30*np.pi*2)

    lines = initialize()

    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    set_axes_equal(ax)

    from matplotlib.animation import FuncAnimation, writers
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, keypoints3d.shape[0]), interval=1000/fps, repeat=False, fargs=[lines])

    Writer = writers['ffmpeg']
    writer = Writer(fps=fps, metadata={}, bitrate=500000)
    anim.save(filename, writer=writer)

    return anim