import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pose_pipeline import TopDownPerson


def center_skeleton(keypoints3d):
    joints = TopDownPerson.joint_names()
    pelvis = np.mean(keypoints3d[:, np.array([joints.index('Left Hip'), joints.index('Right Hip')])], axis=1, keepdims=True)
    centered = keypoints3d - pelvis

    # TODO: this rotation should be removed later
    from scipy.spatial.transform import Rotation as R
    K = R.from_euler('XYZ', np.array([30, 120, 0]), degrees=True).as_matrix()

    return centered @ K


def skeleton_video(keypoints3d, filename, fps=30.0):

    keypoints3d = center_skeleton(keypoints3d)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    joints = TopDownPerson.joint_names()
    left = ['Left Ankle', 'Left Knee', 'Left Hip', 'Left Shoulder', 'Left Elbow', 'Left Wrist']
    left = np.array([joints.index(l) for l in left])
    right = ['Right Ankle', 'Right Knee', 'Right Hip', 'Right Shoulder', 'Right Elbow', 'Right Wrist']
    right = np.array([joints.index(l) for l in right])

    def initialize():
        frame_idx = keypoints3d.shape[0] // 2
        lines = []

        lines.append(plt.plot(keypoints3d[frame_idx, :, 0], keypoints3d[frame_idx, :, 2], -keypoints3d[frame_idx, :, 1], '.'))
        lines.append(plt.plot(keypoints3d[frame_idx, left, 0], keypoints3d[frame_idx, left, 2], -keypoints3d[frame_idx, left, 1], 'b'))
        lines.append(plt.plot(keypoints3d[frame_idx, right, 0], keypoints3d[frame_idx, right, 2], -keypoints3d[frame_idx, right, 1], 'r'))

        return lines

    def update_video(frame_idx, lines):
        lines = lines.copy()

        lines[0][0].set_xdata(keypoints3d[frame_idx, :, 0])
        lines[0][0].set_ydata(keypoints3d[frame_idx, :, 2])
        lines[0][0].set_3d_properties(-keypoints3d[frame_idx, :, 1], zdir='z')

        lines[1][0].set_xdata(keypoints3d[frame_idx, left, 0])
        lines[1][0].set_ydata(keypoints3d[frame_idx, left, 2])
        lines[1][0].set_3d_properties(-keypoints3d[frame_idx, left, 1], zdir='z')

        lines[2][0].set_xdata(keypoints3d[frame_idx, right, 0])
        lines[2][0].set_ydata(keypoints3d[frame_idx, right, 2])
        lines[2][0].set_3d_properties(-keypoints3d[frame_idx, right, 1], zdir='z')

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