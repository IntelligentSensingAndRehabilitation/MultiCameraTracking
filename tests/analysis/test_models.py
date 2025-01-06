import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_load_smpl():
    from easymocap.smplmodel.body_param import load_model
    smpl_path = '/home/isr/app/packages/EasyMocap/data/smplx'
    body_model = load_model(model_path=smpl_path)

    assert body_model is not None, "SMPL Model not loaded correctly"

def test_top_down():
    from multi_camera.utils.standard_pipelines import reconstruction_pipeline
    from pose_pipeline.pipeline import Video, TopDownMethodLookup, PersonBbox, TrackingBboxMethodLookup
    from multi_camera.datajoint.sessions import Recording
    from multi_camera.datajoint.multi_camera_dj import (
        PersonKeypointReconstructionMethodLookup,
        PersonKeypointReconstruction,
        SingleCameraVideo,
        CalibratedRecording,
)

    tracking_method_name = "Easymocap"
    top_down_method_name = "MMPose_RTMPose_Cocktail14"
    reconstruction_method_name = "Implicit Optimization KP Conf, MaxHuber=10"

    filt = PersonKeypointReconstructionMethodLookup * TopDownMethodLookup & {
        "top_down_method_name": top_down_method_name,
        "reconstruction_method_name": reconstruction_method_name,
    }

    annotated = CalibratedRecording & (
            Video * SingleCameraVideo * PersonBbox * TrackingBboxMethodLookup & {"tracking_method_name": tracking_method_name}
        )
    
    keys = ((CalibratedRecording & Recording & annotated - (PersonKeypointReconstruction & filt)) & 'cal_timestamp LIKE "2024-12-17%"').fetch("KEY")
    print(keys)
    reconstruction_pipeline(
        keys,
        top_down_method_name=top_down_method_name,
        reconstruction_method_name=reconstruction_method_name,
        reserve_jobs=True,
    )
