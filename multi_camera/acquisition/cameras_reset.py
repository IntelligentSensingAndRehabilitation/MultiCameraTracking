import simple_pyspin
from simple_pyspin import Camera
import yaml


def reset(num_cams=1, config="", verbose=False):

    # Get the available cameras
    camera_list = simple_pyspin.list_cameras()

    # Check if a config file has been provided
    if config != "":
        with open(config, "r") as file:
            camera_config = yaml.safe_load(file)
        print(f"Selecting cameras defined in {config}.")
        # Get list of all available cameras
        cams = [Camera(i, lock=True) for i in range(camera_list.GetSize())]

    else:
        # Set num_cams to the # of available cameras
        # if input arg is larger (num_cams = min(num_cams,len(camera_list))
        num_cams = min(num_cams,len(camera_list))
        print(f"No config file passed. Selecting the first {num_cams} cameras in the list.")
        # First create list of first n cameras in camera_list where n=num_cams
        cams = [Camera(i, lock=True) for i in range(num_cams)]

    if verbose:
        print(f"List of cams: {cams}")

    for c in cams:
        c.init()


        # check if the current camera is in the list defined by config
        if config != "":
            if int(c.DeviceSerialNumber) not in camera_config["camera-info"].keys():
                if verbose:
                    print(f"{c.DeviceSerialNumber} not listed in config file.")
                continue
            print(f"Reset {c.DeviceSerialNumber}")
            c.DeviceReset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record video from GigE FLIR cameras")
    parser.add_argument("-n", "--num_cams", type=int, default=1, help="Number of input cameras")
    parser.add_argument("-c", "--config", default="", type=str, help="Path to a config.yaml file")
    args = parser.parse_args()

    reset(num_cams=args.num_cams,
          config=args.config)
