"""
Subject-identity extension tables. Schema isolates PHI (e.g., hospital FIN)
from the session/recording data in mocap_sessions. Tables here are
session-keyed via cross-schema FK to mocap_sessions.Session.
"""

import datajoint as dj

from multi_camera.datajoint import sessions

schema = dj.schema("subject_extended")

# Enforced at the API entry point (backend/fastapi.py /session, which mirrors the
# value because importing this module activates the schema) and again in
# sessions.import_session before the push transaction.
FIN_MAX_LENGTH = 20


@schema
class Fin(dj.Manual):
    definition = f"""
    # Hospital FIN for a session (PHI). Row presence implies FIN was provided.
    -> sessions.Session
    ---
    fin: varchar({FIN_MAX_LENGTH})
    """
