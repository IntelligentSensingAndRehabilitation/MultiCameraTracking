from simple_pyspin import Camera


def reset(num_cams=1):
    cams = [Camera(i, lock=True) for i in range(num_cams)]

    print(f"List of cams: {cams}")

    for c in cams:
        c.init()
        print(f"Reset {c.DeviceSerialNumber}")
        c.DeviceReset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record video from GigE FLIR cameras")
    parser.add_argument("-n", "--num_cams", type=int, default=1, help="Number of input cameras")
    args = parser.parse_args()

    reset(num_cams=args.num_cams)
