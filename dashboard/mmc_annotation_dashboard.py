import streamlit as st
import numpy as np
from sensor_fusion.mmc_linkage import RecordingLink
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording
from multi_camera.datajoint.sessions import Recording, Session
from multi_camera.datajoint.annotation import (
    VideoActivityLookup,
    WalkingTypeLookup,
    AssistiveDeviceLookup,
)
import pandas as pd

st.set_page_config(layout="wide",page_title="MMC Annotation Dashboard")

participant_id_options = np.unique(Session.fetch("participant_id"))
participant_id = st.selectbox("Participant ID", participant_id_options)

session_date_options = (Session & {"participant_id": participant_id}).fetch(
    "session_date"
)
session_date = st.selectbox("Session Date", session_date_options)

df = (
    MultiCameraRecording
    * (Recording & ({"participant_id": participant_id, "session_date": session_date}))
).fetch(as_dict=True)

my_column_config = {
    "camera_config_hash": None,
    "session_date": None,
    "recording_timestamps": None,
    "video_project": None,
}
disabled_editing = ["participant_id", "video_base_filename", "comment"]

# add video activity column to df
df = pd.DataFrame(df)
df["Annotation"] = None
df["Annotation Sub-Type"] = None
df["Assistive Device"] = None

# configure column options
my_column_config.update(
    {
        "Annotation": st.column_config.SelectboxColumn(
            options=list(VideoActivityLookup.fetch("video_activity"))
        )
    }
)
my_column_config.update(
    {
        "Annotation Sub-Type": st.column_config.SelectboxColumn(
            options=list(WalkingTypeLookup.fetch("walking_type"))
        )
    }
)
my_column_config.update(
    {
        "Assistive Device": st.column_config.SelectboxColumn(
            options=list(AssistiveDeviceLookup.fetch("assistive_device"))
        )
    }
)


st.data_editor(df, column_config=my_column_config, disabled=disabled_editing)
