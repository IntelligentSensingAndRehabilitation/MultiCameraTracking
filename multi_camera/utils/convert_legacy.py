# Support for importing legacy recordings that split the video top and bottom

import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def make_splits(dual_file_name, num_splits=2, camera_names=None):

    assert os.path.exists(dual_file_name)

    print(f'Splitting into {num_splits}')

    if camera_names is None:
        camera_names = range(num_splits)
    else:
        assert len(camera_names) == num_splits

    file_base = os.path.splitext(dual_file_name)[0]
    filenames = [f'{file_base}.{c}.mp4' for c in camera_names]

    # TODO: consider controlling bitrate with https://stackoverflow.com/questions/38686359/opencv-videowriter-control-bitrate
    vid = cv2.VideoCapture(dual_file_name)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_size = (width, int(height // num_splits))

    writers = [cv2.VideoWriter(fn, fourcc, fps, out_size) for fn in filenames]

    for i in tqdm(range(frames)):
        ret, frame = vid.read()
        images = np.split(frame, num_splits, axis=0)

        for writer, image in zip(writers, images):
            writer.write(image)

    vid.release()

    for writer in writers:
        writer.release()

    return filenames


def grab_preview(vid_name):
    from PIL import Image

    vid = cv2.VideoCapture(vid_name)
    for i in range(10):
        _, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vid.release()

    return Image.fromarray(frame)


def convert_legacy(vid_name, flip=None):

    vid_base = os.path.splitext(vid_name)[0]

    timestamps = json.load(open(os.path.join(vid_name, vid_base + '.json'), 'r'))
    if 'serials' in timestamps.keys():
        serials = timestamps['serials']
        print(f'Serials: {serials}')
        make_splits(vid_name, len(serials), serials)

    else:
        camera_names = ['UnknownRight', 'UnknownLeft']
        assert flip is not None, "Please specify flip direction for videos without serial numbers"
        if flip:
            camera_names.reverse()
        print(camera_names)
        make_splits(vid_name, len(camera_names), camera_names)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Compute calibration from specified videos and insert into database")
    parser.add_argument("vid_name", help="Filename of video to convert to new format")
    args = parser.parse_args()

    convert_legacy(vid_name=args.vid_name)
