import datajoint as dj
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording
import numpy as np

schema = dj.schema('multicamera_tracking_annotation')

@schema
class VideoActivityLookup(dj.Lookup):
    definition = """
    video_activity: varchar(32) # the activity someone is doing in a video
    """
    contents = zip(
        [
            "Overground Walking",
            "Treadmill Walking",
            "Parallel Bar Walking",
            "Tandem Walking",
            "Standing",
            "Sitting",
            "Sit Stand Transition",
            "Stairs",
            "Mixed",
            "Fall",
            "Stumble",
            "FSST",
            "TUG",
            "PST Open",
            "PST Closed",
            "Ramp",
            "cognitive TUG",
            "L-test",
            "Other",
        ]
    )


@schema
class VideoActivity(dj.Manual):
    definition = """
    # annotates the activity done in a recording
    -> MultiCameraRecording
    ---
    -> VideoActivityLookup
    """

    def get_walking(self=None):
        if self is None:
            self = VideoActivity()
        return self & "video_activity IN ('Overground Walking', 'Treadmill Walking', 'Parallel Bar Walking')"

@schema
class WalkingTypeLookup(dj.Lookup):
    definition = """
    walking_type: varchar(32)
    """
    contents = zip(["Fast", "Slow", "ssgs"])

@schema
class WalkingType(dj.Manual):
    definition = """
    # annotates the subtype of walking. This is only for overground walking.
    -> VideoActivity 
    ---
    -> WalkingTypeLookup
    """

    def safe_insert(keys, **kwargs):
        possible_strings = ['Overground Walking']
        activities = np.unique((VideoActivity & keys).fetch("video_activity"))
        assert np.isin(activities, possible_strings).all(), "Only Overground Walking is allowed for this table"
        if len(keys) == 1 or type(keys) == dict:
            WalkingType.insert1(keys, **kwargs)
        else:
            WalkingType.insert(keys, **kwargs)

@schema
class TUGTypeLookup(dj.Lookup):
    definition = """
    tug_type: varchar(32)
    """
    contents = zip(["Normal", "Cognitive"])


@schema
class TUGType(dj.Manual):
    definition = """
    # annotates the subtype of TUG. This is only for TUG.
    -> VideoActivity
    ---
    -> TUGTypeLookup
    """

    def safe_insert(keys, **kwargs):
        possible_strings = ['TUG']
        activities = np.unique((VideoActivity & keys).fetch("video_activity"))
        assert np.isin(activities, possible_strings).all(), "Only TUG is allowed for this table"
        if len(keys) == 1 or type(keys) == dict:
            TUGType.insert1(keys, **kwargs)
        else:
            TUGType.insert(keys, **kwargs)




@schema
class AssistiveDeviceLookup(dj.Lookup):
    definition = """
    assistive_device : varchar(50)
    """
    contents = zip(
        [
            "None",
            "Parallel Bars",
            "Cane Left",
            "Cane Right",
            "Quad Cane Left",
            "Quad Cane Right",
            "Crutch Left",
            "Crutch Right",
            "Bilateral Crutches",
            "Forearm Crutch Left",
            "Forearm Crutch Right",
            "Bilateral Forearm Crutches",
            "Rolling Walker",
            "Harness",
            "Side Walker"
            "Standing Walker"
        ]
    )

@schema
class AssistiveDevice(dj.Manual):
    definition = """
    # annotates the assistive device used in a recording
    -> MultiCameraRecording
    ---
    -> AssistiveDeviceLookup
    """