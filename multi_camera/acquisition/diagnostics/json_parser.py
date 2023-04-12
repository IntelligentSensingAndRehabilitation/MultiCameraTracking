import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pathlib import Path

def parse_json(data_path):
    # get list of each json file in data path
    json_list = [file for file in os.listdir(data_path) if file.endswith('.json')]

    fig_name = data_path.split('\\')[-2]

    # print(data_path.split('\\'))

    delta_from_main_all = []
    end_delta_list = []

    # load each json file
    for json_file in json_list:
        current_json = open(data_path+json_file)
        current_data = json.load(current_json)

        camera_ids = current_data["serials"]
        id_map = {id:i for i,id in enumerate(camera_ids)}
        # print(id_map)

        ts = np.array(current_data["timestamps"])
        # print(ts)
        dt = (ts - ts[0, 0]) / 1e9
        spread = np.max(dt, axis=1) - np.min(dt, axis=1)
        spread1 = np.max(dt, axis=0) - np.min(dt, axis=0)
        # print(dt)
        ts_df = pd.DataFrame(ts / 1e9,columns=camera_ids)
        dt_df = pd.DataFrame(dt,columns=camera_ids)

        delta_from_main = pd.DataFrame((ts[:, 1:] - ts[:, :1]) / 1e9, columns=camera_ids[1:]) * 1000.
        print(delta_from_main.shape)
        delta_from_main_all.extend(delta_from_main.values)
        print(len(delta_from_main_all))
        # print("MAIN")
        # print(delta_from_main)
        dt_df = dt_df.diff(1)
        dt_df = dt_df - np.mean(dt_df)
        max_min_df = pd.DataFrame()
        #cur_max = dt_df.idxmax(axis=1)
        #cur_min = dt_df.idxmin(axis=1)
        #max_min_df['max'] = cur_max
        #max_min_df['min'] = cur_min
        #max_min_df.replace({"max": id_map}, inplace=True)
        #max_min_df.replace({"min": id_map}, inplace=True)
        # print(ts_df)
        # print(dt_df)
        # print(max_min_df)

        # dt_df.plot.line(figsize=(10, 5))
        # plt.title(f"{json_file}: max spread: {round(np.max(spread) * 1000, 2)}")
        print(camera_ids)
        print(delta_from_main.values[-1])
        end_delta_list.append(delta_from_main.values[-1])
        # delta_from_main.plot.line(figsize=(10,5))
        # plt.title(f'delta_from_main, max spread: {round(np.max(spread) * 1000, 2)}')

        # pd.DataFrame(spread).plot.line()

        # fig,axs = plt.subplots(2,1,figsize=(5,10))
        #
        # dt_df.diff(1).plot.line(figsize=(10, 5),ax=axs[0])
        # plt.title(f"{json_file}: max spread: {round(np.max(spread) * 1000,2)}")
        #
        # max_min_df.plot(y=["max", "min"], kind="bar", rot=0,ax=axs[1])
        # axs[1].legend(bbox_to_anchor=(1, 1.02), loc='upper left')
        # axs[1].yaxis.set_major_formatter(StrMethodFormatter("v + {x}"))
        # plt.show()

        # print(f"For {json_file}: Timestamps showed a maximum spread of {np.max(spread) * 1000} ms")
    pd.DataFrame(delta_from_main_all,columns=camera_ids[1:]).plot.line(figsize=(10, 5))
    plt.title(f'delta_from_main')
    plt.ylabel('Time Spread (ms)')
    # pd.DataFrame(delta_from_main_all, columns=camera_ids[1:]).abs().plot.line(figsize=(10, 5))
    # plt.title(f'delta_from_main log')
    # plt.ylabel('Log Time Spread (ms)')
    # # plt.savefig(f"D:\\CottonLab\\multi_camera_data\\timespread_debugging\\full_0302_experiment\\figures\\{fig_name}.png")
    plt.show()

    print(pd.DataFrame(end_delta_list,columns=camera_ids[1:]).tail(3).abs().mean())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse JSON files created from collecting video from GigE FLIR cameras")
    parser.add_argument("data_path", help="Path containing JSON files to parse")

    args = parser.parse_args()

    parse_json(
        data_path=args.data_path,
    )