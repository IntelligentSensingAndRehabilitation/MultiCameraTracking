import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
from pose_pipeline import BlurredVideo
from sensor_fusion.mmc_linkage import RecordingLink
from multi_camera.datajoint.multi_camera_dj import (
    MultiCameraRecording,
    SingleCameraVideo,
)
from multi_camera.datajoint.sessions import Recording, Session
from multi_camera.datajoint.annotation import (
    VideoActivityLookup,
    WalkingTypeLookup,
    AssistiveDeviceLookup,
    VideoActivity as MMCVideoActivity,
    WalkingType as MMCWalkingType,
    AssistiveDevice as MMCAssistiveDevice,
)

# remove old videos
old_videos = glob.glob('tmp*.mp4')
for old_video in old_videos: os.remove(old_video)

st.set_page_config(layout="wide", page_title="MMC Annotation Dashboard")

# choose which subjects/sessions
res = Recording & (MultiCameraRecording & 'video_project NOT LIKE "CUET"') #TODO: need to expand this later
participant_id_options = np.unique((Session & res).fetch("participant_id"))
participant_id = st.selectbox("Participant ID", participant_id_options)

session_date_options = (Session & {"participant_id": participant_id}).fetch(
    "session_date"
)
session_date = st.selectbox("Session Date", session_date_options)

# construct dataframe
df = (
    MultiCameraRecording
    * (Recording & ({"participant_id": participant_id, "session_date": session_date}))
).fetch(as_dict=True)

# get filenames for video display
base_filenames = [x["video_base_filename"] for x in df]

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

def validate_annotations(df):
    for index, row in df.iterrows():
        if row['Annotation Sub-Type'] is not None:
            if row['Annotation'] != 'Overground Walking' and row['Annotation'] is not None:
                print(row['Annotation'] == None)
                print('\n\n\n')
                st.error("Annotation Sub-Type can only be set for Overground Walking. Please correct this.")
                return False
    return True

def clean_annotations(df):
    for index,row in df.iterrows():
        if row['Annotation'] is None and row['Annotation Sub-Type'] in ['Fast','Slow','ssgs']:
            df.loc[index,'Annotation'] = 'Overground Walking'
            st.warning('Inferring Overground Walking from Annotation Sub-Type')
    return df

video_activity_mapping = {
    "fast": "Overground Walking",
    "slow": "Overground Walking",
    "ssgs": "Overground Walking",
    "_ss_": "Overground Walking",
    "tug": "TUG",
    "fsst": "FSST",
    "tandem": "Tandem Walking",
    "eyes closed": "PST Closed",
    "eyes open": "PST Open",
    "TUG": "TUG",
    "PST_closed": "PST Closed",
    "PST_open": "PST Open",
    "PS_closed": "PST Closed",
    "PS_open": "PST Open",
}
walking_type_mapping = {
    "fast": "Fast",
    "slow": "Slow",
    "ssgs": "ssgs",
    "_ss_": "ssgs",
}

def fill_annotation(row, mapping):
    for keyword, value in mapping.items():
        if keyword.lower() in row["video_base_filename"].lower():
            return value
    return None

df["Annotation"] = df.apply(fill_annotation, mapping=video_activity_mapping, axis=1)
df["Annotation Sub-Type"] = df.apply(fill_annotation, mapping=walking_type_mapping, axis=1)


with st.form(key="my_form"):
    st.title("Enter/Edit Annotations")
    edited_df = st.data_editor(df, column_config=my_column_config, disabled=disabled_editing, key="data_editor")

    submitted = st.form_submit_button("Submit")
    if submitted and validate_annotations(edited_df):
        edited_df = clean_annotations(edited_df)
        print(st.session_state['data_editor'])
        for idx,row in edited_df.iterrows():
            if row["Annotation"] is not None:
                key = (MultiCameraRecording & {'recording_timestamps':row['recording_timestamps']}).fetch1("KEY")
                key.update({'video_activity': row["Annotation"]})
                MMCVideoActivity.insert1(key, skip_duplicates=True)
            if row["Annotation Sub-Type"] is not None:
                key = (MMCVideoActivity & {'recording_timestamps':row['recording_timestamps']}).fetch1("KEY")
                key.update({'walking_type': row["Annotation Sub-Type"]})
                MMCWalkingType.safe_insert(key, skip_duplicates=True)
            if row["Assistive Device"] is not None:
                key = (MultiCameraRecording & {'recording_timestamps':row['recording_timestamps']}).fetch1("KEY")
                key.update({'assistive_device': row["Assistive Device"]})
                MMCAssistiveDevice.insert1(key, skip_duplicates=True)
            else:
                key = (MultiCameraRecording & {'recording_timestamps':row['recording_timestamps']}).fetch1("KEY")
                key.update({'assistive_device': "None"})
                MMCAssistiveDevice.insert1(key, skip_duplicates=True)

# visualization
selected_recording = st.selectbox("Select Recording", base_filenames)

if len(BlurredVideo & (SingleCameraVideo & (MultiCameraRecording & {"video_base_filename": selected_recording}))) < 2:
    with st.spinner("Generating blurred videos..."):
        BlurredVideo.populate(SingleCameraVideo & ((SingleCameraVideo & (MultiCameraRecording & {"video_base_filename": selected_recording})) - BlurredVideo).fetch('KEY',limit=2))

single_camera_vids = (BlurredVideo & (
    SingleCameraVideo
    & (MultiCameraRecording & {"video_base_filename": selected_recording})
)).fetch("output_video", limit=2)

col1, col2 = st.columns(2)
try:
    with col1:
        st.video(single_camera_vids[0])
    with col2:
        st.video(single_camera_vids[1])
except:
    st.write("No videos available for this recording")
