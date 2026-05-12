"""
This is the schema to organize recording sessions with the multi camera
recording system. This may be updated in the future with a more comprehensive
refactoring of the multi_camera_dj module and corresponding schema.

For now this is designed so there is only one session on a given date for a
given subject. This could be modified making that field a datetime.

The MultiCameraRecording table doesn't have any type of session organization,
which makes it independent of this experimental organization. This could be
a strength for certain applications, although I think in the future we will
replace this with a dj.Elements design. This table also includes a project
field. In hindsite, this isn't the best place for it. The project identification
probably makes sense more as tags here with the Subject or Session table.
However, even that might not always generalize which is why I didn't make
it explicit as fields in those tables. For example, we may ultimately want
some recordings to be associated with multiple projects. Thus I think it makes
the most sense for this additional information to be added by foreign keys
to other elements.

The Recording table also has a foreign key to the MultiCameraRecording table,
which loosely ensures data integrity through the database. Finally, because
the recordings in a session can be manually inserted into the database as
they are performed, this introduces a slight risk of running an analysis at
the subject or session level that might be invalidated by a later recording.
"""

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Tuple

import datajoint as dj
from .multi_camera_dj import import_recording, MultiCameraRecording

schema = dj.schema("mocap_sessions")


@dataclass
class PhotoSpec:
    saved_path: str
    filename: str
    original_filename: str
    upload_timestamp: datetime
    description: Optional[str] = None


def get_subject_id_from_participant_id(participant_id: str) -> int:
    participant_mapping = {"TF47": 190, "TF02": 191, "TF01": 192}

    if participant_id in participant_mapping:
        return participant_mapping[participant_id]

    # now check it can be cast into an integer and return it or raise an error
    return int(participant_id)


@schema
class Subject(dj.Manual):
    definition = """
    participant_id: varchar(50)
    ---
    """


@schema
class Session(dj.Manual):
    definition = """
    -> Subject
    session_date: date
    ---
    fin=null: varchar(20)   # Hospital FIN for this session (PHI; nullable until provided)
    """

    class Photo(dj.Part):
        definition = """
        # Patient identification photos captured at session start
        -> master
        filename: varchar(255)        # saved filename (timestamp-prefixed, unique within session)
        ---
        photo: attach@localattach     # image file copied into localattach store on insert
        original_filename: varchar(255)
        description=null: varchar(255)
        upload_timestamp: datetime
        """


@schema
class Recording(dj.Manual):
    definition = """
    -> Session
    -> MultiCameraRecording
    ---
    comment: varchar(255)
    """


def import_session(
    participant_id: str,
    session_date: date,
    video_project: str,
    recordings: List[Tuple[str, str]],
    fin: Optional[str] = None,
    photos: Optional[List[PhotoSpec]] = None,
):
    """
    Import a session with a list of recordings

    Args:
        participant_id (str): subject id
        session_date (date): session date
        video_project (str): video project
        recordings (List[Tuple(str, str)]): list of recording tuples
        fin (Optional[str]): hospital FIN for this session
        photos (Optional[List[PhotoSpec]]): patient identification photos for this session

    """

    subject_key = {"participant_id": participant_id}
    session_key = {"participant_id": participant_id, "session_date": session_date}

    assert not Session & session_key, "Session already exists"

    dj.conn().start_transaction()
    try:
        if not Subject & subject_key:
            Subject.insert1(subject_key)

        Session.insert1({**session_key, "fin": fin})

        for recording in recordings:
            print("processing recording", recording)
            vid_base, comment = recording

            vid_dir, vid_base = os.path.split(vid_base)
            print("vid_dir", vid_dir, "vid_base", vid_base)
            key = import_recording(
                vid_base, vid_dir, video_project, skip_connection=True
            )
            key = (MultiCameraRecording & key).fetch1("KEY")
            key.update(session_key)
            key["comment"] = comment

            Recording.insert1(key)

        if photos:
            for p in photos:
                if not os.path.exists(p.saved_path):
                    raise FileNotFoundError(f"Photo missing on disk: {p.saved_path}")
                Session.Photo.insert1({
                    **session_key,
                    "filename": p.filename,
                    "photo": p.saved_path,
                    "original_filename": p.original_filename,
                    "description": p.description,
                    "upload_timestamp": p.upload_timestamp,
                })

    except Exception as e:
        dj.conn().cancel_transaction()
        raise e
    else:
        dj.conn().commit_transaction()
