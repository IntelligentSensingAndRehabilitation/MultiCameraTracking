import os
import time
from concurrent.futures import ProcessPoolExecutor

GPUS = [0, 1, 2, 3, 4, 5, 6] # , 6, 7] # [1, 2, 3, 5, 6, 7]
#GPUS = [1, 3, 4, 5, 6, 7] # , 6, 7] # [1, 2, 3, 5, 6, 7]
#GPUS = [0, 2, 4, 5, 6]
#GPUS = [3,6]
GPUS = [3,4,5,6]
GPUS = [0,1,4,5,6]
GPUS = [4,5]

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def run_pipeline_on_gpu(gpu):
    import os
    import sys
    import pose_pipeline
    from multi_camera.utils import standard_pipelines
    from multi_camera.datajoint.multi_camera_dj import PersonKeypointReprojectionQuality
    from multi_camera.datajoint.biomechanics import BiomechanicalReconstruction
    from multi_camera.datajoint.gaitrite_comparison import (
        GaitRiteSession,
        GaitRiteRecording,
        GaitRiteCalibration,
        GaitRiteRecordingAlignment,
        GaitRiteRecordingStepPositionError,
        GaitRiteRecordingStepLengthError,
        GaitRiteRecordingStepWidthError,
    )
    from multi_camera.datajoint.multi_camera_dj import CalibratedRecording

    pose_pipeline.set_environmental_variables()
    sys.path.append("/home/jcotton/projects/pose/openpose/build/python")

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    pose_pipeline.env.pytorch_memory_limit()
    pose_pipeline.env.tensorflow_memory_limit()

    time.sleep(gpu)

    keys = GaitRiteSession.fetch("KEY")
    keys = (CalibratedRecording & GaitRiteRecording).fetch("KEY")
    #keys = (CalibratedRecording & (GaitRiteRecording & "subject_id IN (192, 313, 316, 321)")).fetch("KEY")
    #keys = (CalibratedRecording & (GaitRiteRecording & "subject_id IN (144)")).fetch("KEY")
    #keys = (CalibratedRecording & (GaitRiteRecording & "subject_id IN (327, 404, 418, 601, 602)")).fetch("KEY")
    #keys = (CalibratedRecording & (GaitRiteRecording & "subject_id IN (601, 602, 603, 604, 605)")).fetch("KEY")
    #keys = (CalibratedRecording & (GaitRiteRecording & "subject_id IN (608)")).fetch("KEY")
    keys = keys[GPUS.index(gpu) :: len(GPUS)]

    print(f"GPU {gpu} found {len(keys)} keys")

    for k in keys:
        print(k)

        # for t in (CalibratedRecording & (GaitRiteRecording & k)).fetch("KEY"):
        for t in [k]:
            # excluding "MMPoseWholebody" because the confidence seems very noisy and
            # thus reconstructions using this are not very good
            keypoints = [
                #"Bridging_bml_movi_87",
                "MMPoseHalpe",
                #"OpenPose",
                #"OpenPose_LR",
                #"OpenPose_HR",
                #"MMPoseWholebody",
                "Bridging_COCO_25",
            ]
            methods = [
                "Robust Triangulation",
                "Implicit Optimization",
                "Explicit Optimization",
                "Explicit Optimization KP Conf, MaxHuber=10",
                "Implicit Optimization KP Conf, MaxHuber=10",
                "Triangulation",
                "Robust Triangulation $\\\\sigma=100$",
                "Robust Triangulation $\\\\sigma=50$",
                "Robust Triangulation $\\\\gamma=0.3$",
                "Implicit Optimization KP Conf",
                "Implicit Optimization $\\\gamma=0.3$",
                "Implicit Optimization, MaxHuber=10",
                "Implicit Optimization $\\\\sigma=50$",
            ]
            
            #keypoints = ["Bridging_bml_movi_87", "MMPoseHalpe"]
            #methods = ["Robust Triangulation", "Implicit Optimization KP Conf, MaxHuber=10"]

            for keypoint in keypoints:
                for method in methods:
                    print(method)
                    standard_pipelines.reconstruction_pipeline(
                        t, top_down_method_name=keypoint, reconstruction_method_name=method
                    )

            PersonKeypointReprojectionQuality.populate(GaitRiteRecording & t)
            GaitRiteCalibration.populate(GaitRiteRecording & t)
            GaitRiteRecordingAlignment.populate(GaitRiteRecording & t)
            GaitRiteRecordingStepPositionError.populate(GaitRiteRecording & t, suppress_errors=True)
            GaitRiteRecordingStepLengthError.populate(GaitRiteRecording & t, suppress_errors=True)
            GaitRiteRecordingStepWidthError.populate(GaitRiteRecording & t, suppress_errors=True)


def process_project():
    from itertools import repeat

    from pose_pipeline.pipeline import schema

    (schema.jobs & 'status="reserved"').delete()
    (schema.jobs & 'status="error"').delete()

    executor = ProcessPoolExecutor(max_workers=len(GPUS))
    res = executor.map(run_pipeline_on_gpu, GPUS)
    for a in res:
        print(a)
        print(a.result())


if __name__ == "__main__":

    process_project()
