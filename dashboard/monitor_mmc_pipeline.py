import streamlit as st
import numpy as np
import pandas as pd

from multi_camera.datajoint.utils.session_stats import get_project_stats_counts, get_stats

from multi_camera.datajoint.sessions import Recording
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording

projects = np.unique((MultiCameraRecording & Recording).fetch("video_project"))

# Cache the stats fetching using st.cache_data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_all_project_stats():
    projects = np.unique((MultiCameraRecording & Recording).fetch("video_project"))
    stats = [get_project_stats_counts(project) for project in projects]
    stats_df = pd.DataFrame(stats)
    stats_df.set_index("Project", inplace=True)
    return stats_df

st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title="MMC Dashboard",  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

# Fetch stats with caching
stats = fetch_all_project_stats()

st.header("Processing stages for the projects")
st.write("Select check box by project and a column to get list of recordings")

h = 20
event = st.dataframe(stats, on_select="rerun", key="data", selection_mode=["multi-row", "multi-column"], height=len(stats) * h, row_height=h)

if len(event.selection.rows) > 0 and len(event.selection.columns) > 0:
    for i in event.selection.rows:
        print(i)
        project = MultiCameraRecording & {'video_project': stats.index[i]}
        project_stats = get_stats(project)
        for j in event.selection.columns:
            st.write("Selected project:", stats.index[i], " and column ", j)
            query = project_stats[j]
            df = pd.DataFrame((MultiCameraRecording * query).fetch(as_dict=True))
            st.dataframe(df)