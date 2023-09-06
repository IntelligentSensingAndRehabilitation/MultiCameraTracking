import trimesh
import torch
import pytorch3d
import pytorch3d.renderer
import pytorch3d.utils
import numpy as np


def pyvista_read_vtp(fn):
    import pyvista

    if ".ply" in fn:
        fn = fn.split(".ply")[0]

    reader = pyvista.get_reader(fn)
    mesh = reader.read()
    mesh = mesh.triangulate()

    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
    tmesh = trimesh.Trimesh(mesh.points, faces_as_array)
    tmesh = tmesh.smoothed()
    return tmesh


def load_skeleton_meshes(skeleton):
    objects = {}

    for b in skeleton.getBodyNodes():
        for s in b.getShapeNodes():
            name = s.getName()
            shape = s.getShape()
            scale = shape.getScale()
            mesh = pyvista_read_vtp(shape.getMeshPath())

            # deferring transforms for now
            # mesh = mesh.apply_transform(s.getWorldTransform().matrix())
            mesh.vertices = mesh.vertices * scale
            objects[name] = mesh

    return objects


def pose_skeleton(skeleton, pose, meshes=None):
    skeleton.setPositions(pose)

    if meshes is None:
        meshes = load_skeleton_meshes(skeleton)

    posed_meshes = {}
    for b in skeleton.getBodyNodes():
        for s in b.getShapeNodes():
            name = s.getName()
            mesh = meshes[name].copy()
            mesh = mesh.apply_transform(s.getWorldTransform().matrix())
            # mesh.vertices = mesh.vertices * 1000
            posed_meshes[name] = mesh

    return posed_meshes


def render_scene(meshes, focal_length=None, height=None, width=None, cameras=None, color=(0, 1, 0), device="cpu"):
    """Render the mesh under camera coordinates
    meshes: list of trimesh.Trimesh
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    """

    pytorch_meshes = []
    for mesh in meshes:
        # Initialize each vertex to be white in color.
        vert = torch.from_numpy(mesh.vertices).float().to(device)
        face = torch.from_numpy(mesh.faces).long().to(device)
        verts_rgb = torch.ones_like(vert)
        texture = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb[None, ...])

        mesh = pytorch3d.structures.Meshes(verts=vert[None, ...], faces=face[None, ...], textures=texture)
        pytorch_meshes.append(mesh)

    mesh = pytorch3d.structures.join_meshes_as_scene(pytorch_meshes)

    # Initialize a camera.
    if cameras is None:
        cameras = pytorch3d.renderer.PerspectiveCameras(
            focal_length=(
                (focal_length, focal_length),
            ),  # ((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
            device=device,
        )

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width), blur_radius=0.0, faces_per_pixel=10, max_faces_per_bin=50000, bin_size=100
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=(color,), diffuse_color=(color,), specular_color=(color,), shininess=64, device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=pytorch3d.renderer.SoftPhongShader(device=device, cameras=cameras, lights=lights, materials=materials),
    )

    # Do rendering
    if len(cameras) > 1:
        imgs = renderer(mesh.extend(len(cameras)))
    else:
        imgs = renderer(mesh)[0]
    return imgs


def render_poses(skeleton, poses, device="cuda:0"):
    """Render centered skeleton

    Args:
        skeleton: nimble.Skeleton already scaled
        poses: np.array, shape (T, n_dofs)

    Returns:
        images: list of np.array, shape (T, H, W, 4) RGBA format
    """

    from tqdm import tqdm

    images = []
    for pose in tqdm(poses):
        posed = pose_skeleton(skeleton, pose)

        scene = trimesh.Scene()
        for k, v in posed.items():
            scene.add_geometry(v, node_name=k)

        origin = np.mean(posed["pelvis_ShapeNode_0"].vertices, axis=0)
        offset = np.array([0, 0, 8]) - origin * 1.0

        scene.apply_translation(offset)
        objs = scene.dump()

        img = render_scene(objs, 8, 640, 640, color=(0.5, 0.5, 0.5), device=device).cpu().detach().numpy()
        images.append(img)

    return images


def write_images_to_video(outfile, images, fps=30):
    import cv2

    h, w, _ = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outfile, fourcc, fps, (w, h))
    for img in images:
        if img.dtype != np.uint8:
            img = 255 * img[..., :3]
            img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

    out.release()


def get_skeleton_mesh_overlay(key, cam_idx=0):
    """
    Get the function to overlay the skeleton mesh on the image

    Args:
        key: dict, the key to the MultiCameraRecording table
        cam_idx: int, the camera index

    Returns:
        overlay: function, the function to overlay the skeleton mesh on the image

    Note: this uses both jax and pytorch so should make sure neither consumes
    all the GPU memory
    """

    from jaxlie import SO3
    from jax import vmap, numpy as jnp

    from multi_camera.analysis.camera import get_intrinsic, get_extrinsic, distort_3d
    from multi_camera.analysis.biomechanics import bilevel_optimization

    from pose_pipeline.pipeline import VideoInfo, TopDownPerson
    from multi_camera.datajoint.calibrate_cameras import Calibration
    from multi_camera.datajoint.multi_camera_dj import (
        MultiCameraRecording,
        SingleCameraVideo,
        PersonKeypointReconstruction,
    )
    from multi_camera.validation.biomechanics.biomechanics import BiomechanicalReconstruction

    # set up the cameras
    camera_params, camera_names = (Calibration & key).fetch1("camera_calibration", "camera_names")

    videos = (TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key).proj()
    video_keys, video_camera_name = (TopDownPerson * SingleCameraVideo * videos).fetch("KEY", "camera_name")
    assert camera_names == video_camera_name.tolist(), "Videos don't match cameras in calibration"

    width = int(np.unique((VideoInfo & video_keys).fetch("width"))[0])
    height = int(np.unique((VideoInfo & video_keys).fetch("height"))[0])
    N = camera_params["tvec"].shape[0]

    def conv(x):
        return torch.from_numpy(np.array(x))

    camera_matrix = conv(vmap(get_intrinsic, (None, 0), 0)(camera_params, np.arange(N)))

    def rotation_matrix(camera_params, i):
        rvec = jnp.take(camera_params["rvec"], i, axis=0)
        rot = SO3.exp(rvec)
        return rot.as_matrix()

    R = conv(vmap(rotation_matrix, (None, 0), 0)(camera_params, np.arange(N)))

    image_size = conv((np.ones((N, 2)) * np.array([height, width])).astype(int))

    cameras = pytorch3d.utils.cameras_from_opencv_projection(R, conv(camera_params["tvec"]), camera_matrix, image_size)

    # select the desired one since we are only supporting a single camera for now
    cameras = cameras[cam_idx]

    # get the skeleton
    model_name, skeleton_def = (BiomechanicalReconstruction & key).fetch1("model_name", "skeleton_definition")
    skeleton = bilevel_optimization.reload_skeleton(model_name, skeleton_def["group_scales"])
    timestamps, poses = (BiomechanicalReconstruction.Trial & key).fetch1("timestamps", "poses")

    meshes = load_skeleton_meshes(skeleton)

    # match timestamps
    vid_timestamps = (VideoInfo & video_keys[0]).fetch_timestamps()
    frame_idx = np.intersect1d(vid_timestamps, timestamps, return_indices=True)[1]
    assert len(frame_idx) == len(timestamps)
    frame_idx = frame_idx.tolist()

    def overlay(frame, idx):
        try:
            pose_idx = frame_idx.index(idx)
        except ValueError:
            return frame

        p = poses[pose_idx]
        posed = pose_skeleton(skeleton, p, meshes)

        # use trimesh to compose the scene. account for the different coordinate convention
        # between nimblephysics and our camera system
        scene = trimesh.Scene()
        for k, v in posed.items():
            v = v.copy()
            vertices = v.vertices[:, [2, 0, 1]]
            # account for camera distortion. convert vertices to mm first.
            vertices_distorted = np.array(distort_3d(camera_params, cam_idx, vertices * 1000.0))
            # now transform back to world coordinates
            extri = get_extrinsic(camera_params, cam_idx)
            vertices_distorted = np.concatenate(
                [vertices_distorted, np.ones((*vertices_distorted.shape[:-1], 1))], axis=-1
            )
            # transform the points into the camera perspective
            # last dimension ensures broadcasting works
            transformed = (np.linalg.inv(extri) @ vertices_distorted[..., None])[..., 0]
            # drop the last dimension
            vertices_distorted = transformed[..., :-1]
            # then back to meters
            vertices_distorted = vertices_distorted / 1000.0

            v.vertices = vertices_distorted
            scene.add_geometry(v)

        objs = scene.dump()

        with torch.no_grad():
            img = (
                render_scene(objs, height=height, width=width, cameras=cameras.to("cuda:0"), device="cuda:0")
                .cpu()
                .detach()
                .numpy()
            )

        alpha = img[..., -1:]
        frame2 = frame / 255.0 * (1 - alpha) + img[..., :-1] * alpha
        frame2[frame2 > 1.0] = 1.0
        frame2 = (frame2 * 255).astype(np.uint8)

        return frame2

    return overlay


def get_markers_overlay(key, cam_idx=0, radius=5, color=(0, 0, 255)):
    """
    Get the overlay function for the markers

    Args:
        key: dict, the key for the biomechanical reconstruction
        cam_idx: int, the index of the camera to use
        radius: int, the radius of the markers
        color: tuple, the color of the markers

    Returns:
        overlay: function, the overlay function
    """

    from pose_pipeline.pipeline import VideoInfo, TopDownPerson
    from multi_camera.datajoint.multi_camera_dj import (
        MultiCameraRecording,
        SingleCameraVideo,
        PersonKeypointReconstruction,
    )
    from multi_camera.datajoint.calibrate_cameras import Calibration
    from multi_camera.validation.biomechanics.biomechanics import BiomechanicalReconstruction

    from pose_pipeline.utils.visualization import draw_keypoints
    from multi_camera.analysis.camera import project_distortion
    from .bilevel_optimization import get_markers, reload_skeleton

    # set up the cameras
    camera_params, camera_names = (Calibration & key).fetch1("camera_calibration", "camera_names")

    # find the videos (to get the height and width)
    videos = (TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key).proj()
    video_keys, video_camera_name = (TopDownPerson * SingleCameraVideo * videos).fetch("KEY", "camera_name")
    assert camera_names == video_camera_name.tolist(), "Videos don't match cameras in calibration"

    # get video parameters
    width = np.unique((VideoInfo & video_keys).fetch("width"))[0]
    height = np.unique((VideoInfo & video_keys).fetch("height"))[0]

    # load the skeleton
    model_name, skeleton_def = (BiomechanicalReconstruction & key).fetch1("model_name", "skeleton_definition")
    skeleton = reload_skeleton(model_name, skeleton_def["group_scales"])
    timestamps, poses = (BiomechanicalReconstruction.Trial & key).fetch1("timestamps", "poses")

    # get the markers
    markers = get_markers(skeleton, skeleton_def, poses, original_format=True)
    markers = np.stack(list(markers.values()), axis=1)  # drop the names
    markers = markers * 1000.0  # convert to mm expected by the camera library

    # project the markers into the image plane
    markers_proj = project_distortion(camera_params, cam_idx, markers)
    # append ones to the end to indicate high confidence
    markers_proj = np.concatenate([markers_proj, np.ones((*markers_proj.shape[:-1], 1))], axis=-1)

    # handle any bad projections off the edge of the frame
    clipped = np.logical_or.reduce(
        (
            markers_proj[..., 0] <= 0,
            markers_proj[..., 0] >= width,
            markers_proj[..., 1] <= 0,
            markers_proj[..., 1] >= height,
        )
    )
    markers_proj[clipped, 0] = 0
    markers_proj[clipped, 1] = 0
    markers_proj[clipped, 2] = 0

    # match timestamps
    vid_timestamps = (VideoInfo & video_keys[0]).fetch_timestamps()
    frame_idx = np.intersect1d(vid_timestamps, timestamps, return_indices=True)[1]
    assert len(frame_idx) == len(timestamps)
    frame_idx = frame_idx.tolist()

    def overlay(image, idx):
        try:
            pose_idx = frame_idx.index(idx)
        except ValueError:
            return image

        return draw_keypoints(image, markers_proj[pose_idx], radius=radius, color=color)

    return overlay


def create_centered_video(key, out_file_name=None):
    import cv2
    import os
    import tempfile
    from .bilevel_optimization import get_markers, reload_skeleton
    from multi_camera.validation.biomechanics.biomechanics import BiomechanicalReconstruction
    from pose_pipeline.utils.video_format import compress

    model_name, skeleton_def = (BiomechanicalReconstruction & key).fetch1("model_name", "skeleton_definition")
    skeleton = reload_skeleton(model_name, skeleton_def["group_scales"], return_map=False)
    poses = (BiomechanicalReconstruction.Trial & key).fetch1("poses")

    images = render_poses(skeleton, poses)

    fd, file_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    write_images_to_video(file_name, images)
    compressed = compress(file_name)
    os.remove(file_name)

    if out_file_name is not None:
        os.rename(compressed, out_file_name)
    else:
        out_file_name = compressed

    return out_file_name


def create_overlay_video(key, cam_idx, out_file_name=None):
    import cv2
    import os
    import tempfile
    from pose_pipeline.utils.visualization import video_overlay
    from pose_pipeline.pipeline import BlurredVideo, TopDownPerson
    from multi_camera.utils.visualization import get_projected_keypoint_overlay
    from multi_camera.datajoint.multi_camera_dj import (
        MultiCameraRecording,
        SingleCameraVideo,
        PersonKeypointReconstruction,
    )

    mesh_overlay = get_skeleton_mesh_overlay(key, cam_idx)
    markers_overlay = get_markers_overlay(key, cam_idx, radius=7)
    keypoints_overlay = get_projected_keypoint_overlay(key, cam_idx, radius=7)

    def overlay(image, idx):
        image = mesh_overlay(image, idx)
        image = keypoints_overlay(
            image,
            idx,
        )
        image = markers_overlay(image, idx)
        return image

    if out_file_name is None:
        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

    # find the videos (to get the height and width)
    videos = (TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key).proj()
    video_keys, video_camera_name = (TopDownPerson * SingleCameraVideo * videos).fetch("KEY", "camera_name")

    video_key = video_keys[cam_idx]

    video = (BlurredVideo & video_key).fetch1("output_video")
    video_overlay(video, out_file_name, overlay, max_frames=None, downsample=1, compress=True)

    os.remove(video)

    return out_file_name
