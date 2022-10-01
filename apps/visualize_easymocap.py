import os

from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, CalibratedRecording, SMPLReconstruction
from multi_camera.datajoint.easymocap import EasymocapTracking, EasymocapSmpl

import easymocap
from easymocap.socket.base_client import BaseSocketClient
from easymocap.socket.o3d import VisOpen3DSocket
from easymocap.config.vis_socket import Config


def stream_easymocap_key(key, smpl=False, filter_subjects=None):

    easymocap_dir = os.path.split(os.path.split(easymocap.__file__)[0])[0]
    if smpl:
        os.chdir(easymocap_dir) # required as it looks for link yml in relative directory
        cfg = Config.load(os.path.join(easymocap_dir, 'config/vis3d/o3d_scene_smpl.yml'))
        results = (EasymocapSmpl & key).fetch1('smpl_results')
    else:
        cfg = Config.load(os.path.join(easymocap_dir, 'config/vis3d/o3d_scene.yml'))
        results = (EasymocapTracking & key).fetch1('tracking_results')

    print(f'Fetched data for {key}. Starting server')

    cfg['host'] = '127.0.0.1'
    cfg['post'] = '9990'

    server = VisOpen3DSocket(cfg.host, cfg.port, cfg)
    server.update()

    # 2. set the ip address and port
    client = BaseSocketClient(cfg.host, cfg.port)

    for r in results:
        if filter_subjects is not None:
            r = [p for p in r if p['id'] in filter_subjects]

        if smpl:
            client.send_smpl(r)
        else:
            client.send(r)
        server.update()

    client.close()

    exit()


def stream_smpl_key(key):

    easymocap_dir = os.path.split(os.path.split(easymocap.__file__)[0])[0]
    os.chdir(easymocap_dir) # required as it looks for link yml in relative directory
    cfg = Config.load(os.path.join(easymocap_dir, 'config/vis3d/o3d_scene_smpl.yml'))
    results = (EasymocapSmpl & key).fetch1('smpl_results')

    results = (SMPLReconstruction & key).fetch('poses', 'shape', 'orientation', 'translation', as_dict=True)[0]

    def restructure_frame(results, i):
        res = {'id': 0,
            'poses': results['poses'][None, i],
            'shapes': results['shape'],
            'Rh': results['orientation'][None, i],
            'Th': results['translation'][None, i]}
        return res

    smpl_results = [[restructure_frame(results, i)] for i in range(results['poses'].shape[0])]
    print(f'Fetched data for {key}. Starting server')

    cfg['host'] = '127.0.0.1'
    cfg['post'] = '9990'

    server = VisOpen3DSocket(cfg.host, cfg.port, cfg)
    server.update()

    # 2. set the ip address and port
    client = BaseSocketClient(cfg.host, cfg.port)

    for r in smpl_results:
        client.send_smpl(r)
        server.update()

    client.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Fetch Easymocap data from MultiCamera DJ and visualize")
    parser.add_argument("vid_base", help="Base filenames to use for calibration")
    parser.add_argument("--smpl", help="Use Easymocap SMPL", action='store_true')
    parser.add_argument("--top_down", help="Use top down SMPL", action='store_true')
    parser.add_argument("--select_cal", help="Select amongst calibration", default=None)
    parser.add_argument("--filter", help="Filter by subject ID", default=None)
    args = parser.parse_args()

    if args.filter is not None:
        filter = args.filter.split(',')
        filter = [int(f) for f in filter]
    else:
        filter = None

    if args.top_down and args.smpl:

        key = (SMPLReconstruction * MultiCameraRecording * CalibratedRecording & f'video_base_filename="{args.vid_base}"').fetch('KEY')
        if len(key) > 1 and args.select_cal != None:
            key = key[int(args.select_cal)]
            print(f'Selected key: {key}')
        elif len(key) > 1:
            raise Exception('Multiple calibrations matched this. Please use --select_cal')
        else:
            key = key[0]

        stream_smpl_key(key)

    else:

        key = (MultiCameraRecording * CalibratedRecording & f'video_base_filename="{args.vid_base}"').fetch('KEY')
        if len(key) > 1 and args.select_cal != None:
            key = key[int(args.select_cal)]
            print(f'Selected key: {key}')
        elif len(key) > 1:
            raise Exception('Multiple calibrations matched this. Please use --select_cal')
        else:
            key = key[0]

        stream_easymocap_key(key, args.smpl, filter)
