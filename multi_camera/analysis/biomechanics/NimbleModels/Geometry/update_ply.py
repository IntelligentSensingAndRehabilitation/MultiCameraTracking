import argparse
import os
import pyvista

if __name__ == '__main__':

    # Parse a vtp file and convert it to a ply file
    parser = argparse.ArgumentParser(description='Convert a folder of vtp files to a folder of ply files')
    parser.add_argument('src_folder', help='folder containing the vtp files to convert', default="./", type=str)
    parser.add_argument('dst_folder', help='folder to save the ply files', default="./", type=str)

    args = parser.parse_args()

    src = args.src_folder
    if args.dst_folder is None:
        target = src + "../Geometry_ply/"
    else:
        target = args.dst_folder
        if target[-1] != "/":
            target += "/"

    os.makedirs(target, exist_ok=True)

    # for each file in src
    for filename in os.listdir(src):

        ext = os.path.splitext(filename)[-1]
        if ext not in [".vtp", ".obj"]:
            print("Skipping " + filename)
            continue

        reader = pyvista.get_reader(src + filename)

        mesh = reader.read()
        mesh = mesh.triangulate()
        # mesh.plot()

        mesh.save(target + filename + ".ply")
        print("Converted mesh: " + target + filename + ".ply")