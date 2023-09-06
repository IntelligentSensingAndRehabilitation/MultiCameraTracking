from typing import List

# TODO: create server configuration for things like this
model_path: str = "/Mocap/model_data/smpl_clean/"


def get_unannotated_recordings():
    from multi_camera.datajoint.easymocap import EasymocapSmpl
    from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo
    from multi_camera.datajoint.sessions import Recording
    from pose_pipeline.pipeline import PersonBbox

    keys = (EasymocapSmpl * MultiCameraRecording * Recording - SingleCameraVideo * PersonBbox).fetch("KEY")
    video_base_filenames = (MultiCameraRecording & keys).fetch("video_base_filename")
    return video_base_filenames


def get_mesh(filename: str, downsampling: int = 5):
    print("Mesh fetch requested for ", filename, downsampling)
    from multi_camera.datajoint.easymocap import EasymocapSmpl
    from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording
    from easymocap.smplmodel.body_model import SMPLlayer
    import concurrent.futures
    import base64
    import numpy as np
    from tqdm import tqdm

    smpl_results = (EasymocapSmpl & (MultiCameraRecording & {"video_base_filename": filename})).fetch1("smpl_results")
    smpl_results = smpl_results[::downsampling]

    # create layer to process the meshes
    smpl = SMPLlayer(model_path, model_type="smpl", gender="neutral")

    # get the unique list of ids
    ids = [[r["id"] for r in frame] for frame in smpl_results]
    ids = list(set([id for frame in ids for id in frame]))

    # use concurrent futures to process batches of 10 frames in parallel
    concurrent_workers = 100

    def process_frame(frame, smpl):
        smpl_fields = ["poses", "shapes", "Rh", "Th"]
        smpl_batch = {}
        for f in smpl_fields:
            smpl_batch[f] = np.concatenate([person[f] for person in frame], axis=0)

        ids = [int(person["id"]) for person in frame]

        verts = smpl(**smpl_batch, return_verts=True, return_tensor=False)
        verts = (verts * 1000.0).astype(int)

        def encode(v):
            return v

        return [{"id": i, "verts": v.tolist()} for i, v in zip(ids, verts)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = [executor.submit(process_frame, frame, smpl) for frame in tqdm(smpl_results)]
        # now collect gather the futures results
        results = [f.result() for f in tqdm(futures)]

    def encode(v):
        import json

        return base64.b64encode(json.dumps(v).encode("utf-8")).decode("utf-8")

    result = {
        "ids": [int(x) for x in ids],
        "frames": len(smpl_results),
        "type": "smpl",
        "faces": smpl.faces.astype(int).tolist(),
        "meshes": encode(results),
    }

    return result


def annotate_recording(filename: str, ids: List[int]):
    from multi_camera.datajoint.easymocap import EasymocapTracking
    from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording

    match = EasymocapTracking & (MultiCameraRecording & {"video_base_filename": filename})

    if len(match) == 1:
        print(f"Annotating {filename} with {ids} (match: {match})")
        match.create_bounding_boxes(ids)

    return True
