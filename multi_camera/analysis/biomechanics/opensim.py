import numpy as np
import pandas as pd


def joint_renamer(j):
    j = j.replace('Sternum', 'Neck')
    j = j.replace('Right ', 'R')
    j = j.replace('Left ', 'L')
    j = j.replace('Little', 'Small')
    j = j.replace('Pelvis', 'CHip')
    j = j.replace(' ', '')
    return j


def normalize_marker_names(joints):
    """ Convert joint names to those expected by OpenSim model """
    return [joint_renamer(j) for j in joints]


def points3d_to_trc(points3d, filename, marker_names, fps=30):
    '''
    Exports a set of points into an OpenSim TRC file

    Modified from Pose2Sim.make_trc

    Parameters:
        points3d (np.array) : time X joints X 3 array
        filename (string) : file to export to
        marker_names (list of strings) : names of markers to annotate
        fps : frame rate of points
    '''

    assert len(marker_names) == points3d.shape[1], "Number of marker names must match number of points"
    f_range = [0, points3d.shape[0]]

    # flatten keypoints after reordering axes
    points3d = np.take(points3d, [1, 2, 0], axis=-1)
    points3d = points3d.reshape([points3d.shape[0], -1])

    #Header
    DataRate = CameraRate = OrigDataRate = fps
    NumFrames = points3d.shape[0]
    NumMarkers = len(marker_names)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + filename,
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames',
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, f_range[0], f_range[1]])),
            'Frame#\tTime\t' + '\t\t\t'.join(marker_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(marker_names))])]

    #Add Frame# and Time columns
    Q = pd.DataFrame(points3d)
    Q.insert(0, 't', Q.index / fps)

    #Write file
    with open(filename, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, line_terminator='\n')

    return Q