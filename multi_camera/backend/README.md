# Markerless Mocap System Backend

This serves as a REST backend for both the acquisition system and the datajoint schema, as well
as bridging between the two. That means it allows running experiments, which are tracked in a
local database and migrating these to DataJoint. It also enables working with the DJ results,
such as annotating the recordings.

`rest_backend.py` is the main entry point for the backend. It is a FastAPI that serves the
acquisition system.

