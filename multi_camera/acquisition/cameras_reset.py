import simple_pyspin
from simple_pyspin import Camera
import yaml


def reset(all_cams=True, config="", verbose=False):

    # Get the available cameras
    camera_list = simple_pyspin.list_cameras()
    cams = [Camera(i, lock=True) for i in range(camera_list.GetSize())]

    if verbose:
        print(f"Total cams: {len(cams)}")
        print(f"List of cams: {cams}")

    # Check if either flag has been provided (config or all cams)
    if config == "" and not all_cams:
        print(f"Please specify which cameras to reset. Exiting.")
        return
    elif config != "":
        with open(config, "r") as file:
            camera_config = yaml.safe_load(file)
        print(f"Selecting cameras defined in {config}.")
    else:
        # Reset all available cameras
        print(f"No config file passed. Resetting {len(cams)} discovered cameras.")

    for i, c in enumerate(cams):
        try:
            c.init()

            # check if the current camera is in the list defined by config
            if config != "":
                if int(c.DeviceSerialNumber) not in camera_config["camera-info"].keys():
                    if verbose:
                        print(f"{c.DeviceSerialNumber} not listed in config file.")
                    continue

            c.DeviceReset()
            print(f"Reset {(i+1):02d}) {c.DeviceSerialNumber}")
            del c
        except Exception as E:
            print(E)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reset video from GigE FLIR cameras")
    parser.add_argument("-a", "--all_cams", default=False, action="store_true", help="Reset all discovered cameras")
    parser.add_argument("-c", "--config", default="", type=str, help="Path to a config.yaml file")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Control verbosity of code")
    args = parser.parse_args()

    reset(all_cams=args.all_cams, config=args.config, verbose=args.verbose)
