import os
import cv2
import torch
import numpy as np
from tqdm import trange
from torchvision import models
from torch.cuda.amp import autocast as autocast
from pose_pipeline.env import add_path
from pose_pipeline.utils.bounding_box import crop_image_bbox

checkpoint_path = '/home/jcotton/projects/pose/MvMHAT/models/model_20220905_080000.pth'

def mvmhat(video, tracks, return_extra=False):
    '''
    Run MvMHAT on a list of videos with a set of bounding box tracks

    Parameters:
        videos : list of strings to video filenames
        tracks : list of bounding box tracks for each video in format from PosePipeline

    Returns:
        Dictionary of combined tracks. Each dictionary entry corresponds to a
        view.
    '''

    with add_path(os.environ["MVMHAT_PATH"]):
        from deep_sort.update import Update
        from deep_sort.mvtracker import MVTracker

        model = models.resnet50(pretrained=False)
        model = model.cuda()
        ckp = torch.load(checkpoint_path)['model']
        model.load_state_dict(ckp)
        model.eval()

        def process_video(video, tracks):
            ''' Compute ReID for each of the bounding boxes in tracks '''

            cap = cv2.VideoCapture(video)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            #frames = 100
            #tracks = tracks[:frames]

            tracks_out = tracks.copy()

            for i in trange(frames):
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                for b in range(len(tracks[i])):
                    cropped = crop_image_bbox(frame, tracks[i][b]['tlhw'])[0]

                    with torch.no_grad():
                        img = torch.Tensor(cropped.transpose([2, 0, 1])[None, ...]).cuda()
                        with autocast():
                            features = model(img).detach().cpu().numpy().tolist()

                    tracks_out[i][b]['features'] = features[0]

            cap.release()

            return tracks_out

        def tracks_to_mvmhat(tracks_out):
            '''Convert tracks to format for MvMHAT'''

            # convert data to the format used by their library
            seq_dict = {}
            for view_idx in range(len(tracks_out)):

                det = []
                for frame_idx in range(len(tracks_out[view_idx])):
                    for bbox_idx in range(len(tracks_out[view_idx][frame_idx])):
                        track =  tracks_out[view_idx][frame_idx][bbox_idx]
                        det.append([frame_idx] + [bbox_idx] + track['tlhw'].tolist() + [1] + [0, 0, 0] + track['features'])

                seq_dict[view_idx] = {
                    "sequence_name": 'test',
                    "image_filenames": None, #image_filenames[view],
                    "detections": np.array(det),
                    "groundtruth": None, #groundtruth,
                    "image_size": (3, 1520, 2704),
                    "min_frame_idx": 0, #dataset_info['start'],
                    "max_frame_idx": frame_idx, #dataset_info['end'] - 1,
                    "feature_dim": 1000,
                    "update_ms": 10
                }

            return seq_dict

        featured_tracks = [process_video(video[i], tracks[i]) for i in range(len(video))]
        seq_dict = tracks_to_mvmhat(featured_tracks)

        mvtracker = MVTracker(list(range(0, len(tracks))))
        updater = Update(seq=seq_dict, mvtracker=mvtracker, display=False)
        updater.run()

        if return_extra:
            return updater.result, {'updater': updater, 'mvtracker': mvtracker,
                                    'tracks': featured_tracks, 'seq_dict': seq_dict}
        return updater.result