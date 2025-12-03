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
            "Wheelchair Propulsion",
            "L-test"
            "FACIAL_ROM_BrowsRest",
            "FACIAL_ROM_BrowsRaised",
            "FACIAL_ROM_FurrowBrow",
            "FACIAL_ROM_EyesOpen",
            "FACIAL_ROM_EyesWideOpen",
            "FACIAL_ROM_EyesClosed",
            "FACIAL_ROM_LipsRestClosed",
            "FACIAL_ROM_MouthClosed",
            "FACIAL_ROM_Smile",
            "FACIAL_ROM_Pucker",
            "FACIAL_ROM_PuckerStretch",
            "FACIAL_ROM_MouthWideOpen",
            "FMS",
            "FMS_ActiveStraightLegRaise",
            "FMS_AnkleClearingTest",
            "FMS_DeepSquat",
            "FMS_HurdleStep",
            "FMS_InlineLunge",
            "FMS_RotaryStability",
            "FMS_ShoulderClearingTest",
            "FMS_ShoulderMobility",
            "FMS_SingleLegSquat",
            "FMS_SpinalExtensionClearingTest",
            "FMS_SpinalFlexionClearingTest",
            "FMS_TrunkStabilityPushUp",
            "CUET",
            "CUET_2FingerPinch",
            "CUET_3FingerPinch",
            "CUET_AcquireRelease",
            "CUET_Container",
            "CUET_Grasp",
            "CUET_LateralPinch",
            "CUET_LiftUp",
            "CUET_PullPush",
            "CUET_PushDown",
            "CUET_ReachDown",
            "CUET_ReachForward",
            "CUET_ReachUp",
            "CUET_WristUp",
            "CUET_ManipulateChip",
            "CUET_Calculator",
            "CUET_Phone",
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
    activity_side = NULL : enum('Left', 'Right', 'Both') # the side of the activity
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
    contents = zip(["Fast", "Slow", "ssgs", "FGA_20ft", "FGA_no", "FGA_yes", "FGA_varying", "FGA_pivot", "FGA_step_over", "FGA_closed", "FGA_backwards"])

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

@schema
class RangeAnnotation(dj.Manual):
    definition = """
    # annotates the range of a recording
    -> MultiCameraRecording
    range_id    :  smallint
    ---
    user        : varchar(32)   # user who annotated the range
    source      : varchar(32)   # data source used while annotating
    ranges      : longblob      # list of ranges in the recording, e.g. [[0, 100], [200, 300]]
    num_ranges  : int unsigned  # number of ranges in the recording
    range_label : varchar(512)  # label for the range, e.g. 'Full', 'Left', 'Right'
    annotation_time : datetime  # time when the annotation was made
    """

@schema
class EventAnnotation(dj.Manual):
    definition = """
    # annotates the range of a recording
    -> MultiCameraRecording
    event_id    :  smallint
    ---
    user        : varchar(32)    # user who annotated the event
    source      : varchar(32)   # data source used while annotating
    events      : longblob       # list of events in the recording, e.g. [[0, 100], [200, 300]]
    num_events  : int unsigned   # number of events in the recording
    event_label : varchar(512)   # label for the event, e.g. 'Left Arm Raise'
    annotation_time : datetime   # time when the annotation was made
    """