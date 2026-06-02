"""
Subject-identity extension tables. Schema isolates PHI (e.g., hospital FIN)
from the session/recording data in mocap_sessions. Tables here are
session-keyed via cross-schema FK to mocap_sessions.Session.
"""

import datajoint as dj

from multi_camera.datajoint import sessions

schema = dj.schema("subject_extended")


@schema
class Fin(dj.Manual):
    definition = """
    # Hospital FIN for a session (PHI). Row presence implies FIN was provided.
    -> sessions.Session
    ---
    fin: varchar(20)
    """
