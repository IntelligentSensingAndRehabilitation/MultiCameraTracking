import datajoint as dj

schema = dj.schema("multicamera_tracking")


@schema
class Calibration(dj.Manual):
    definition = """
    # Calibration of multiple camera system
    cal_timestamp        : timestamp
    camera_config_hash   : varchar(10)
    ---
    num_cameras          : int
    camera_names         : longblob   # list of camera names
    camera_calibration   : longblob   # calibration results
    reprojection_error   : float
    """

def import_recording(vid_base, vid_path='.', video_project='MULTICAMERA_TEST'):
    import os
    import json
    import numpy as np
    import datajoint as dj
    from datetime import datetime

    from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo

    # search for files. expects them to be in the format vid_base.serial_number.mp4
    vids = []
    camera_names = []
    for v in os.listdir(vid_path):
        base, ext = os.path.splitext(v)
        if ext == '.mp4' and base.split('.')[0] == vid_base:
            vids.append(os.path.join(vid_path, v))

    print(f'Found {len(vids)} videos.')

    def mysplit(x):
        splits = x.split('_')
        base = '_'.join(splits[:-2])
        date = '_'.join(splits[-2:])

        return base, date

    camera_names = [os.path.split(v)[1].split('.')[1] for v in vids]
    camera_hash = hex(hash(tuple(np.sort(camera_names)))).split('x')[1][:10]
    _, timestamp = mysplit(vid_base)
    timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')

    parent = {'recording_timestamps': timestamp, 'camera_config_hash': camera_hash, 'video_project': video_project,
              'video_base_filename': vid_base, 'num_cameras': len(vids), 'camera_names': camera_names}

    timestamps = json.load(open(os.path.join(vid_path, vid_base + '.json'), 'r'))
    serials = timestamps['serials']
    frame_timestamps = np.array(timestamps['timestamps'])

    assert all(np.sort(serials) == np.sort(camera_names))

    vid_structs = []
    single_structs = []
    for v, serial in zip(vids, camera_names):

        vid_filename = os.path.split(v)[1]

        vid_struct = {'video_project': video_project, 'filename': vid_filename,
                      'start_time': timestamp, 'video': v}

        ts_idx = serials.index(serial)
        single_struct = {'recording_timestamps': timestamp, 'camera_config_hash': camera_hash, 'camera_name': serial,
                         'video_project': video_project, 'filename': vid_filename, 'frame_timestamps': 0} #list(frame_timestamps[:, ts_idx])}

        vid_structs.append(vid_struct)
        single_structs.append(single_struct)

    dj.conn().start_transaction()
    try:
        MultiCameraRecording.insert1(parent)
        Video.insert(vid_structs)
        SingleCameraVideo.insert(single_structs)
    except Exception as e:
        dj.conn().cancel_transaction()
        raise e
    else:
        dj.conn().commit_transaction()
