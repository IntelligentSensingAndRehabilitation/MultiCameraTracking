import simple_pyspin
from simple_pyspin import Camera
import yaml


def reset(all_cams=True, config="", verbose=False):
    if all_cams:
        import PySpin

        system = PySpin.System.GetInstance()

        cams = system.GetCameras()

        N = cams.GetSize()
        print(f"Resetting {N} cameras.")

        def reset_cam(i):
            """Reset one camera by index; return (serial_or_label, error_or_None)."""
            try:
                c = cams[i]
                serial = c.TLDevice.DeviceSerialNumber.GetValue()
            except Exception as e:
                return f"index={i}", f"could not read serial: {e}"
            try:
                c.Init()
                c.DeviceReset()
                c.DeInit()
                print(f"{i}: Reset {serial}")
                del c
                return serial, None
            except Exception as e:
                return serial, str(e)

        import concurrent.futures

        # list(...) forces consumption so per-camera failures surface
        # instead of disappearing into the executor.
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
            results = list(executor.map(reset_cam, range(N)))

        failures = [(s, err) for s, err in results if err is not None]
        for serial, err in failures:
            print(f"  ! Failed to reset {serial}: {err}")

        cams.Clear()

        system.ReleaseInstance()

        if failures:
            print(
                f"Completed with {len(failures)} of {N} cameras failed to "
                "reset. Power-cycle those cameras manually if they remain "
                "unresponsive."
            )
        else:
            print("Completed resetting all cameras. Exiting.")
        return

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
