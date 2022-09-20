import os
import cv2
import torch
import numpy as np
from tqdm import trange
from torchvision import models
from torch.cuda.amp import autocast as autocast
from pose_pipeline.env import add_path
from pose_pipeline.utils.bounding_box import crop_image_bbox

checkpoint_path = '/home/jcotton/projects/pose/MvMHAT/models/model_20220917_154500.pth'
yolov7_pretrained = '/home/jcotton/projects/pose/MultiCameraTracking/notebooks/wrappers/yolov7-w6.pt'


def convert_to_posepipe(view_results, total_frames):
    results = []

    for i in range(total_frames):
        frame_detections = [{'track_id': v[1], 'tlhw': v[2:], 'tlbr': [v[2], v[3], v[2]+v[4], v[3]+v[5]]}
                            for v in view_results if v[0] == i]
        results.append(frame_detections)
    return results


def mvmhat(videos, return_extra=False, max_frames=None, conf_thres=0.5):
    '''
    Run MvMHAT on a list of videos with a set of bounding box tracks

    Parameters:
        videos : list of strings to video filenames
        tracks : list of bounding box tracks for each video in format from PosePipeline

    Returns:
        Dictionary of combined tracks. Each dictionary entry corresponds to a
        view.
    '''

    with add_path(os.environ["YOLOV7_PATH"]):
        from models.yolo import Model
        from utils.general import non_max_suppression

        #yolov7 = attempt_load(yolov7_pretrained, map_location='cuda')
        model = torch.load(yolov7_pretrained)
        if isinstance(model, dict):
            model = model['ema' if model.get('ema') else 'model']

        yolov7 = Model(model.yaml).to(next(model.parameters()).device)  # create
        yolov7.load_state_dict(model.float().state_dict())  # load state_dict
        yolov7.names = model.names  # class names
        #if autoshape:
        #    hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        yolov7.to('cuda')
        yolov7.eval()
        print('YOLOv7 loaded')

    with add_path(os.environ["MVMHAT_PATH"]):
        from deep_sort.update import Update
        from deep_sort.mvtracker import MVTracker

        model = models.resnet50()
        model = model.cuda()
        ckp = torch.load(checkpoint_path)['model']
        model.load_state_dict(ckp)
        model.eval()
        print('MvMHAT ReID loaded')

    def tlbr_to_tlhw (x):
        return [x[0], x[1], x[2]-x[0], x[3]-x[1]]

    def process_video(video):
        cap = cv2.VideoCapture(video)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            frames = max_frames

        dets = []

        for i in trange(frames):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                img = torch.Tensor(frame.transpose([2, 0, 1])).to('cuda').unsqueeze(0) / 255.0
                pred, _ = yolov7(img)

            pred = non_max_suppression(pred, classes=0, conf_thres=conf_thres)[0].detach().cpu().numpy()

            for j, p in enumerate(pred):
                tlhw = tlbr_to_tlhw(p)

                # my library code nicely keeps a square aspect ratio. not what the model is trained on.
                # cropped = crop_image_bbox(frame, np.array(tlhw), target_size=(224, 224), dilate=1.0)[0]

                bbox = np.array(tlhw).astype(int)
                cropped = frame[bbox[1]:bbox[3] + bbox[1], bbox[0]:bbox[2] + bbox[0], :]
                if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    continue

                cropped = cv2.resize(cropped, (224, 224))

                with torch.no_grad():
                    img = torch.Tensor(cropped.transpose([2, 0, 1])[None, ...]).cuda()
                    features = model(img).detach().cpu().numpy().tolist()[0]

                dets.append([i] + [j] + tlhw + [p[4]] + [0, 0, 0] + features)

        cap.release()

        return np.array(dets), frames, frame.shape

    seq_dict = {}
    for view_idx, vid in enumerate(videos):
        dets, frames, shape = process_video(vid)

        seq_dict[view_idx] = {
            "sequence_name": vid,
            "detections": np.array(dets),
            "image_size": (shape[2], *shape[:2]),
            "min_frame_idx": 0,
            "max_frame_idx": frames,
            "feature_dim": 1000,
            "update_ms": 10
        }

    mvtracker = MVTracker(list(range(0, len(videos))))
    updater = Update(seq=seq_dict, mvtracker=mvtracker, display=False)
    updater.run()

    results = [convert_to_posepipe(r, total_frames=seq_dict[k]['max_frame_idx'])
               for k, r in updater.result.items()]

    if return_extra:
        return results, {'updater': updater, 'mvtracker': mvtracker, 'seq_dict': seq_dict}

    return results