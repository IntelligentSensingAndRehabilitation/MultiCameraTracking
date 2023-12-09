import streamlit as st
import numpy as np
import pandas as pd

from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording
from multi_camera.datajoint.annotation import VideoActivity as MMCVideoActivity, WalkingType as MMCWalkingType, AssistiveDevice as MMCAssistiveDevice
from sensor_fusion.session_annotations import VideoActivity as SFVideoActivity, WalkingType as SFWalkingType, AssistiveDevice as SFAssistiveDevice
from sensor_fusion.emgimu_session import FirebaseSession

st.set_page_config(layout="wide", page_title="Annotation Tracker")

# sensor fusion annotations
sf_missing_video_activity = len((FirebaseSession.AppVideo - SFVideoActivity))
sf_missing_walking_type = len(((SFVideoActivity & 'video_activity="Overground Walking"') - SFWalkingType))
sf_missing_assistive_device = len(FirebaseSession.AppVideo - SFAssistiveDevice)

st.write("# Missing Sensor Fusion Annotations")

vp,counts = np.unique((FirebaseSession.AppVideo - SFVideoActivity).fetch("video_project"), return_counts=True)
disp = {}
# display in streamlit
for i in range(len(vp)):
    disp[vp[i]] = counts[i]
# add row called "Video activity" to pandas
df = pd.DataFrame(disp, index=["Video Activity"])

# do the same for walking type
vp,counts = np.unique(((SFVideoActivity & 'video_activity="Overground Walking"') - SFWalkingType).fetch("video_project"), return_counts=True)
disp = {}
# display in streamlit
for i in range(len(vp)):
    disp[vp[i]] = counts[i]
# add walking type row to df
df = pd.concat([df, pd.DataFrame(disp, index=["Walking Type"])], ignore_index=False)

# do the same for assistive device
vp,counts = np.unique((FirebaseSession.AppVideo - SFAssistiveDevice).fetch("video_project"), return_counts=True)
disp = {}
# display in streamlit
for i in range(len(vp)):
    disp[vp[i]] = counts[i]
# add assistive device row to df
df = pd.concat([df, pd.DataFrame(disp, index=["Assistive Device"])], ignore_index=False)

# missing sensor annotations
from sensor_fusion.session_annotations import SessionSensor,SensorPlacement
vp,counts = np.unique(((FirebaseSession.AppVideo & SessionSensor) - SensorPlacement).fetch("video_project"), return_counts=True)
disp = {}
# display in streamlit
for i in range(len(vp)):
    disp[vp[i]] = counts[i]
# add row called "Video activity" to pandas
df = pd.concat([df, pd.DataFrame(disp, index=["Sensor Placement"])], ignore_index=False)


st.dataframe(df)


# multi camera annotations
mmc_missing_video_activity = len((MultiCameraRecording - MMCVideoActivity))
mmc_missing_walking_type = len(((MMCVideoActivity & 'video_activity="Overground Walking"') - MMCWalkingType))
mmc_missing_assistive_device = len(MultiCameraRecording - MMCAssistiveDevice)

st.write("# Multi Camera Annotations")
st.write("Missing Video Activity: ", mmc_missing_video_activity)
st.write("Missing Walking Type: ", mmc_missing_walking_type)
st.write("Missing Assistive Device: ", mmc_missing_assistive_device)



# check if log.csv exists
import os
import pandas as pd
import datetime
import time

if not os.path.exists('log.pkl'):
    # create empty dataframe
    df = pd.DataFrame(columns=['time', 'sf_missing_video_activity', 'sf_missing_walking_type', 'sf_missing_assistive_device', 'mmc_missing_video_activity', 'mmc_missing_walking_type', 'mmc_missing_assistive_device'])
else:
    df = pd.read_pickle('log.pkl')

entry = {'time': datetime.datetime.now(),
            'sf_missing_video_activity': sf_missing_video_activity,
            'sf_missing_walking_type': sf_missing_walking_type,
            'sf_missing_assistive_device': sf_missing_assistive_device,
            'mmc_missing_video_activity': mmc_missing_video_activity,
            'mmc_missing_walking_type': mmc_missing_walking_type,
            'mmc_missing_assistive_device': mmc_missing_assistive_device}
# append to dataframe
df = pd.concat([df, pd.DataFrame(entry, index=[0])], ignore_index=True)

df.to_pickle('log.pkl')
st.line_chart(df,x='time')

while True:
    time.sleep(3600)
    st.experimental_rerun()