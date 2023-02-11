import trimesh
import torch
import pytorch3d
import pytorch3d.renderer
import numpy as np


def pyvista_read_vtp(fn):
    import pyvista
    
    if '.ply' in fn:
        fn = fn.split('.ply')[0]
        
    reader = pyvista.get_reader(fn)
    mesh = reader.read()
    mesh = mesh.triangulate()
    
    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:] 
    tmesh = trimesh.Trimesh(mesh.points, faces_as_array) 
    return tmesh


def load_skeleton_meshes(skeleton):

    objects = {}

    for b in skeleton.getBodyNodes():
        for s in b.getShapeNodes():
            name = s.getName()
            shape = s.getShape()
            mesh = pyvista_read_vtp(shape.getMeshPath())

            # deferring transforms for now
            #mesh = mesh.apply_transform(s.getWorldTransform().matrix())
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
            mesh = meshes[name]
            mesh = mesh.apply_transform(s.getWorldTransform().matrix())
            posed_meshes[name] = mesh

    return posed_meshes


def render_scene(meshes, focal_length, height, width, device='cpu'):
    ''' Render the mesh under camera coordinates
    meshes: list of trimesh.Trimesh
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''

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
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
        device=device,
    )

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),   # (H, W)
        # image_size=height,   # (H, W)
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs


def render_poses(skeleton, poses, device='cuda:0'):
    meshes = load_skeleton_meshes(skeleton)

    images = []
    for pose in poses:

        posed = pose_skeleton(skeleton, pose)

        scene = trimesh.Scene()
        for k, v in posed.items():
            scene.add_geometry(v)

        origin = np.mean(posed['pelvis_ShapeNode_0'].vertices, axis=0)
        offset = np.array([0, 0, 8]) - origin * 1.0

        scene.apply_translation(offset)
        objs = scene.dump()
        posed_meshes = pose_skeleton(skeleton, pose, meshes)

        img = render_scene(objs, 2000, 640, 640, device=device).cpu().detach().numpy()[0]
        images.append(img)

    return images


def write_images_to_video(outfile, images, fps=30):
    import cv2

    h, w, _ = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outfile, fourcc, fps, (w, h))

    for img in images:
        img = 255 * img[..., :3]
        img = img.astype(np.uint8)
        out.write(img)

    out.release()