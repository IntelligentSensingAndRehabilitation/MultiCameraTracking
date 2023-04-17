import os
from pose_pipeline import Video
from pose_pipeline.utils.standard_pipelines import bottom_up_pipeline

from pose_pipeline.pipeline import PersonBbox
from multi_camera.datajoint.sessions import Recording 
from multi_camera.datajoint.gaitrite_comparison import GaitRiteRecording
from multi_camera.datajoint.easymocap import EasymocapSmpl
from multi_camera.datajoint.multi_camera_dj import *

# doesn't matter as library in installed path
os.environ["OPENPOSE_PATH"] = ""

# first perform what is needed for annotation
keys = (Video & (EasymocapSmpl * MultiCameraRecording * Recording - SingleCameraVideo * PersonBbox)).fetch('KEY')
print(len(keys))
bottom_up_pipeline(keys, bottom_up_method_name ="OpenPose_HR", reserve_jobs=False)

# then process everything else we might need
keys = (Video & (SingleCameraVideo * GaitRiteRecording)).fetch('KEY')
keys = (Video & (SingleCameraVideo * Recording)).fetch('KEY') + keys
print(len(keys))
bottom_up_pipeline(keys, bottom_up_method_name ="OpenPose_HR", reserve_jobs=False)

keys = (Video & (SingleCameraVideo * GaitRiteRecording)).fetch('KEY')
keys = (Video & (SingleCameraVideo * Recording)).fetch('KEY') + keys
print(len(keys))
bottom_up_pipeline(keys, bottom_up_method_name ="OpenPose_LR", reserve_jobs=False)
