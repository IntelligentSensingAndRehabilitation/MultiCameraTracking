import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_json(data_path):
    # get list of each json file in data path
    json_list = [file for file in os.listdir(data_path) if file.endswith('.json')]
    # load each json file
    for json_file in json_list:
        current_json = open(data_path+json_file)
        current_data = json.load(current_json)

        camera_ids = current_data["serials"]

        ts = np.array(current_data["timestamps"])
        # print(ts)
        dt = (ts - ts[0, 0]) / 1e9
        spread = np.max(dt, axis=1) - np.min(dt, axis=1)
        # print(dt)
        dt_df = pd.DataFrame(dt,columns=camera_ids)
        # print(dt_df)
        dt_df.diff(10).plot.line(figsize=(10, 5))
        plt.title(f"{json_file}")
        plt.show()

        break
        print(f"For {json_file}: Timestamps showed a maximum spread of {np.max(spread) * 1000} ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse JSON files created from collecting video from GigE FLIR cameras")
    parser.add_argument("data_path", help="Path containing JSON files to parse")

    args = parser.parse_args()

    parse_json(
        data_path=args.data_path,
    )