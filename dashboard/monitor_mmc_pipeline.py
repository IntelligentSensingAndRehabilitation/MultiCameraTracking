import streamlit as st
import numpy as np
import pandas as pd

from pose_pipeline.pipeline import TopDownPerson, TopDownMethodLookup, BottomUpPeople, BottomUpMethodLookup
from multi_camera.datajoint.sessions import Session, Recording
from multi_camera.datajoint.multi_camera_dj import (
    MultiCameraRecording,
    SingleCameraVideo,
    CalibratedRecording,
    PersonKeypointReconstruction,
)
from multi_camera.datajoint.easymocap import EasymocapSmpl

bottom_up = BottomUpPeople * BottomUpMethodLookup & {"bottom_up_method_name": "Bridging_OpenPose"}
top_down = TopDownPerson * TopDownMethodLookup & {"top_down_method_name": "Bridging_bml_movi_87"}

projects = np.unique((MultiCameraRecording & Recording).fetch("video_project"))
missing = (MultiCameraRecording & Recording - CalibratedRecording).proj()

recording = Recording * MultiCameraRecording.proj()

project_filt = lambda project: MultiCameraRecording & {"video_project": project}

stats = [
    {
        "Project": project,
        "Number of recordings": len(Recording & project_filt(project)),
        "Number of videos missing bottom up": len((Recording * SingleCameraVideo - bottom_up) & project_filt(project)),
        "Number of videos missing top down": len(Recording * SingleCameraVideo - top_down & project_filt(project)),
        "Number missing calibration": len(missing & project_filt(project)),
        "Calibrated awaiting initial reconstruction": len(
            (Recording & CalibratedRecording - EasymocapSmpl) & project_filt(project) - PersonKeypointReconstruction
        ),
        "Number without reconstruction": len(
            (Recording & CalibratedRecording - PersonKeypointReconstruction)
            & (MultiCameraRecording & {"video_project": project})
        ),
    }
    for project in projects
]
stats = pd.DataFrame(stats)
stats.set_index("Project", inplace=True)

st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title="MMC Dashboard",  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

st.dataframe(stats, use_container_width=True, height=500)
