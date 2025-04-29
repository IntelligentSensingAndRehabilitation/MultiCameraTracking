import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
from pose_pipeline import BlurredVideo

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
    TUGType as MMCTUGType,
    TUGTypeLookup as MMCTUGTypeLookup,
    AssistiveDevice as MMCAssistiveDevice,
)
import base64

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# remove old videos
old_videos = glob.glob('tmp*.mp4')
for old_video in old_videos: os.remove(old_video)

st.set_page_config(layout="wide", page_title="MMC Annotation Dashboard")


def load_data():
    res = (Recording & (MultiCameraRecording & 'video_project NOT LIKE "h36m"') & 'participant_id NOT LIKE 111') - MMCVideoActivity
    st.session_state['recordings_to_annotate'] = len(res)
    st.session_state['participant_id_options'] = np.unique((Session & res).fetch("participant_id"))
    if 'participant_id' not in st.session_state:
        st.session_state['participant_id'] = st.session_state['participant_id_options'][0] if st.session_state['participant_id_options'].any() else None
    st.session_state['session_date_options'] = (Session & ((Recording & {"participant_id":  st.session_state['participant_id']}) - (MMCVideoActivity & {"participant_id":  st.session_state['participant_id']}))).fetch('session_date')
    if 'session_date' not in st.session_state:
        st.session_state['session_date'] = st.session_state['session_date_options'][0] if st.session_state['session_date_options'].any() else None

if 'initialized' not in st.session_state:
    load_data()
    st.session_state['initialized'] = True

tab1, tab2 = st.tabs(['Annotate', 'View'])

with tab1:
    # choose which subjects/sessions
    res = Recording & (MultiCameraRecording  & 'video_project NOT LIKE "h36m"') - MMCVideoActivity #TODO: need to expand this later
    st.write("# Annotate MMC Videos")
    participant_id = st.selectbox("Participant ID", st.session_state['participant_id_options'], key='participant_id',on_change=load_data)
    participant_res = Recording & {'participant_id':participant_id}# this is necessary to restrict the MultiCameraRecording table to the selected participant
    st.session_state['session_date_options'] = (Session & ((Recording & {"participant_id":  participant_id}) - (MMCVideoActivity & {"participant_id":  participant_id}))).fetch('session_date')
    session_date = st.selectbox("Session Date", st.session_state['session_date_options'], key='session_date',index=0)

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
    df["Activity Side"] = None

    # configure column options
    my_column_config.update(
        {
            "Annotation": st.column_config.SelectboxColumn(
                options=list(VideoActivityLookup.fetch("video_activity"))
            )
        }
    )
    print(VideoActivityLookup.fetch('video_activity'))
    my_column_config.update(
        {
            "Annotation Sub-Type": st.column_config.SelectboxColumn(
                options=list(WalkingTypeLookup.fetch("walking_type")) + list(MMCTUGTypeLookup.fetch("tug_type"))
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
    my_column_config.update(
        {
            "Activity Side": st.column_config.SelectboxColumn(
                options=['Left', 'Right', 'Both']
            )
        }
    )

    def validate_annotations(df):
        for index, row in df.iterrows():            
            if row['Annotation Sub-Type'] is not None:
                if row['Annotation'] not in ['Overground Walking','TUG'] and row['Annotation'] is not None:
                    print(row['Annotation'] == None)
                    print('\n\n\n')
                    st.error("Annotation Sub-Type can only be set for Overground Walking and TUG. Please correct this.")
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
        "6MWT": "Overground Walking",
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
        "LTest": "L-test",
        "ltest": "L-test",
        "l test": "L-test",
        "_FGA_": "Overground Walking",
        "CUET_1_": "CUET_ReachForward",
        "CUET_2_": "CUET_ReachUp",
        "CUET_3_": "CUET_ReachDown",
        "CUET_4_": "CUET_LiftUp",
        "CUET_5_": "CUET_PushDown",
        "CUET_6_": "CUET_WristUp",
        "CUET_7_": "CUET_AcquireRelease",
        "CUET_8_": "CUET_Grasp",
        "CUET_9_": "CUET_LateralPinch",
        "CUET_10_11_": "CUET_PullPush",
        "CUET_10-11_": "CUET_PullPush",
        "CUET_12_": "CUET_Container",
        "CUET_13_": "CUET_2FingerPinch",
        "CUET_14_": "CUET_3FingerPinch",
        "CUET_15_": "CUET_ManipulateChip",
        "CUET_16_": "CUET_Calculator",
        "CUET_17_": "CUET_Phone",
        "circuit": "Mixed",
        "circuits": "Mixed",
        'deep_squat': 'FMS_DeepSquat',
        'hurdle': 'FMS_HurdleStep',
        'hurdle_step': 'FMS_HurdleStep',
        'lunge': 'FMS_InlineLunge',
        'in_line_lunge': 'FMS_InlineLunge',
        'ankle_clearing': 'FMS_AnkleClearingTest',
        'shoulder_mob': 'FMS_ShoulderMobility',
        'shoulder_mobilty': 'FMS_ShoulderMobility',
        'shoulder_mobility': 'FMS_ShoulderMobility',
        'active_slr': 'FMS_ActiveStraightLegRaise',
        'active_straight_leg_raise': 'FMS_ActiveStraightLegRaise',
        'trunk_stability_push_up': 'FMS_TrunkStabilityPushUp',
        'trunk_stabilty_push_up': 'FMS_TrunkStabilityPushUp',
        'stability_pushup': 'FMS_TrunkStabilityPushUp',
        'extension_clearing': 'FMS_SpinalExtensionClearingTest',
        'ext_clearing': 'FMS_SpinalExtensionClearingTest',
        'lumbar_ext_clearing': 'FMS_SpinalExtensionClearingTest',
        'flexsion_clearing': 'FMS_SpinalFlexionClearingTest',
        'flexion_clearing': 'FMS_SpinalFlexionClearingTest',
        'flex_clearing': 'FMS_SpinalFlexionClearingTest',
        'lumbar_flex_clearing': 'FMS_SpinalFlexionClearingTest',
        'rotary_stability': 'FMS_RotaryStability',
        'rotary': 'FMS_RotaryStability',
        'single_leg_squat': 'FMS_SingleLegSquat',
        'sigle_leg_squat': 'FMS_SingleLegSquat',
        'shoulder_clearing': 'FMS_ShoulderClearingTest',
        "brows_rest": "FACIAL_ROM_BrowsRest",
        "brows_raised": "FACIAL_ROM_BrowsRaised",
        "furrow_brow": "FACIAL_ROM_FurrowBrow",
        "eyes_open": "FACIAL_ROM_EyesOpen",
        "eyes_wide_open": "FACIAL_ROM_EyesWideOpen",
        "eyes_closed": "FACIAL_ROM_EyesClosed",
        "lips_rest_closed": "FACIAL_ROM_LipsRestClosed",
        "mouth_closed": "FACIAL_ROM_MouthClosed",
        "smile": "FACIAL_ROM_Smile",
        # "pucker": "FACIAL_ROM_Pucker",
        # "pucker_stretch": "FACIAL_ROM_PuckerStretch",
        "mouth_wide_open": "FACIAL_ROM_MouthWideOpen",

    }
    activity_subtype_mapping = {
        "fast": "Fast",
        "slow": "Slow",
        "ssgs": "ssgs",
        "_ss_": "ssgs",
        "_cogTUG_":"Cognitive",
        "_TUG_":"Normal",
        "FGA_20ft_": "FGA_20ft",
        "FGA_no_": "FGA_no",
        "FGA_yes_": "FGA_yes",
        "FGA_varying_": "FGA_varying",
        "FGA_pivot_": "FGA_pivot",
        "FGA_step_over_": "FGA_step_over",
        "FGA_closed_": "FGA_closed",
        "FGA_backwards_": "FGA_backwards",
        "6MWT": "6MWT",
        "single_leg_squat": "Single Leg Stand",
        "deep_squat":"Deep Squat",
        "hurdle":"Hurdle",
        "lunge":"Lunge",
        "shoulder_mob":"Shoulder Mobility",
        "active_slr":"Active Straight Leg Raise",
        "stability_pushup":"Trunk Stability Push-Up",
        "rotary":"Rotary",
        "ankle_clearing":"Ankle Clearing",
        "shoulder_clearing":"Shoulder Clearing",
        "ext_clearing":"Extension Clearing",
        "flex_clearing":"Flexion Clearing",
    }
    activity_side_mapping = {
        "_L_": "Left",
        "_R_": "Right",
        "left": "Left",
        "right": "Right",
        "both": "Both",
        "bilateral": "Both",
    }

    def fill_annotation(row, mapping):
        for keyword, value in mapping.items():
            if (
                keyword.lower() in row["video_base_filename"].lower()
                or keyword.lower() in row["comment"].lower()
            ):
                return value
        return None

    df["Annotation"] = df.apply(fill_annotation, mapping=video_activity_mapping, axis=1)
    df["Annotation Sub-Type"] = df.apply(fill_annotation, mapping=activity_subtype_mapping, axis=1)
    df["Activity Side"] = df.apply(fill_annotation, mapping=activity_side_mapping, axis=1)

    with st.form(key="my_form"):
        st.title("Enter/Edit Annotations")
        edited_df = st.data_editor(df, column_config=my_column_config, disabled=disabled_editing, key="data_editor")

        submitted = st.form_submit_button("Submit")
        if submitted and validate_annotations(edited_df):
            edited_df = clean_annotations(edited_df)
            print(st.session_state['data_editor'])
            for idx,row in edited_df.iterrows():
                if row["Annotation"] is not None:
                    key = (MultiCameraRecording & participant_res & {'recording_timestamps':row['recording_timestamps']}).fetch1("KEY")
                    key.update({'video_activity': row["Annotation"]})
                    key.update({'activity_side': row["Activity Side"]})
                    MMCVideoActivity.insert1(key, skip_duplicates=True)
                if row["Annotation Sub-Type"] is not None:
                    if row["Annotation"] == "Overground Walking":
                        key = (MMCVideoActivity & participant_res & {'recording_timestamps':row['recording_timestamps']}).fetch1("KEY")
                        key.update({'walking_type': row["Annotation Sub-Type"]})
                        MMCWalkingType.safe_insert(key, skip_duplicates=True)
                    elif row["Annotation"] == "TUG":
                        key = (MMCVideoActivity & participant_res & {'recording_timestamps':row['recording_timestamps']}).fetch1("KEY")
                        key.update({'tug_type': row["Annotation Sub-Type"]})
                        MMCTUGType.safe_insert(key, skip_duplicates=True)
                if row["Assistive Device"] is not None:
                    key = (MultiCameraRecording & participant_res & {'recording_timestamps':row['recording_timestamps']}).fetch1("KEY")
                    key.update({'assistive_device': row["Assistive Device"]})
                    MMCAssistiveDevice.insert1(key, skip_duplicates=True)
                else:
                    key = (MultiCameraRecording & participant_res & {'recording_timestamps':row['recording_timestamps']}).fetch1("KEY")
                    key.update({'assistive_device': "None"})
                    MMCAssistiveDevice.insert1(key, skip_duplicates=True)
            load_data()
            st.success("Annotations successfully submitted")
            st.rerun()

    # visualization
    selected_recording = st.selectbox("Select Recording", base_filenames)

    generate = st.button("Generate Blurred Videos")
    if generate:
        if len(BlurredVideo & (SingleCameraVideo & (MultiCameraRecording & {"video_base_filename": selected_recording}))) < 1:
            with st.spinner("Generating blurred videos..."):
                print(selected_recording)
                BlurredVideo.populate(SingleCameraVideo & ((SingleCameraVideo & (MultiCameraRecording & {"video_base_filename": selected_recording})) - BlurredVideo).fetch('KEY',limit=1))

    single_camera_vids = (BlurredVideo & (
        SingleCameraVideo
        & (MultiCameraRecording & {"video_base_filename": selected_recording})
    )).fetch("output_video", limit=2)

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.video(single_camera_vids[0])
        except:
            st.write("No videos available for this recording")
    with col2:
        try:
            displayPDF(f'/home/jd/projects/experiment_sheets/{participant_id}_{session_date.strftime("%Y%m%d")}.pdf')
        except Exception as e:
            st.write("No PDF available for this recording")

with tab2:
    # choose which subjects/sessions
    st.write("# View MMC Annotations")
    st.write("No editing here, just viewing")
    res = Recording & (MultiCameraRecording & 'video_project NOT LIKE "h36m"')
    participant_id_options= np.unique((Session & res).fetch("participant_id"))
    participant_id = st.selectbox("Participant ID", participant_id_options, key='participant_id2')
    date_options  = (Session & ((Recording & {"participant_id":  participant_id}))).fetch('session_date')
    session_date = st.selectbox("Session Date", date_options)

    # construct dataframe
    df = (
        MultiCameraRecording
        * (Recording & ({"participant_id": participant_id, "session_date": session_date}))
    ).fetch(as_dict=True)

    # get filenames for video display
    base_filenames = [x["video_base_filename"] for x in df]

    # add VideoActivity col to df
    df = pd.DataFrame(df)
    df["Video Activity"] = None
    df["Walking Type"] = None
    df["Assistive Device"] = None
    df["Activity Side"] = None
    for row in df.iterrows():
        try:
            va = (MMCVideoActivity & {'recording_timestamps':row[1]['recording_timestamps']}).fetch1("video_activity")
            df.loc[row[0], "Video Activity"] = va

            if va == "Overground Walking":
                wt = (MMCWalkingType & {'recording_timestamps':row[1]['recording_timestamps']}).fetch1("walking_type")
                df.loc[row[0], "Walking Type"] = wt
            
            ad = (MMCAssistiveDevice & {'recording_timestamps':row[1]['recording_timestamps']}).fetch1("assistive_device")
            df.loc[row[0], "Assistive Device"] = ad

            a_side = (MMCVideoActivity & {'recording_timestamps':row[1]['recording_timestamps']}).fetch1("activity_side")
            df.loc[row[0], "Activity Side"] = a_side
        except Exception as e:
            pass

    my_column_config = {
        "camera_config_hash": None,
        "session_date": None,
        # "recording_timestamps": None,
        "video_project": None,
        "participant_id": None,
    }
    st.dataframe(df,hide_index=True,column_config=my_column_config)

    selected_recording = st.selectbox("Select Recording", base_filenames)

    generate = st.button(" Generate Blurred Videos")
    if generate:
        if len(BlurredVideo & (SingleCameraVideo & (MultiCameraRecording & {"video_base_filename": selected_recording}))) < 1:
            with st.spinner("Generating blurred videos..."):
                print(selected_recording)
                BlurredVideo.populate(SingleCameraVideo & ((SingleCameraVideo & (MultiCameraRecording & {"video_base_filename": selected_recording})) - BlurredVideo).fetch('KEY',limit=1))

    single_camera_vids = (BlurredVideo & (
        SingleCameraVideo
        & (MultiCameraRecording & {"video_base_filename": selected_recording})
    )).fetch("output_video", limit=2)

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.video(single_camera_vids[0])
        except:
            st.write("No videos available for this recording")
    with col2:
        try:
            displayPDF(f'/home/jd/projects/experiment_sheets/{participant_id}_{session_date.strftime("%Y%m%d")}.pdf')
        except Exception as e:
            st.write("No PDF available for this recording")
