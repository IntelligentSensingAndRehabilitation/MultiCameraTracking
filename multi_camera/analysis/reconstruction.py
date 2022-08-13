import numpy as np


def triangulate_with_occlusion(p : np.array, cgroup, threshold=0.25) -> np.array:
    '''
    Triangulate a single point in an occlusion aware manner

        Parameters:
            p (np.array) : C x 3 array of 2D keypoints from C cameras
            cgroup : calibrated camera group

        Returns:
            3 vector
    '''

    visible = p[:, 2] > threshold
    idx = list(np.where(visible)[0])
    if len(idx) == 0:
        return np.nan * np.ones((3))

    cgroup_subset = cgroup.subset_cameras(idx)
    return cgroup_subset.triangulate(p[np.array(idx), None, :2])[0]


def reconstruct(points2d, cgroup, threshold=0.25):
    '''
    Reconstruct 2D keypoints in 3D, accounting for confidence

    Note that this expects the order of the points in the array to
    match the order of the cameras.

        Parameters:
            points2d (np.array) : time X cameras X joints X 3 array
            cgroup (CameraGroup) : calibrated camera group
            threshold (float, option) : minimum confidence to consider a keypoint

        Returns:
            a time X joints x 3 array of (X, Y, Z) coordinates
    '''

    from einops import rearrange
    points2d_flat = rearrange(points2d, 'n c j i -> (n j) c i')
    points3d_flat = np.array([triangulate_with_occlusion(p, cgroup, threshold) for p in points2d_flat])
    points3d = rearrange(points3d_flat, '(n j) i -> n j i', j=points2d.shape[2])

    return points3d
