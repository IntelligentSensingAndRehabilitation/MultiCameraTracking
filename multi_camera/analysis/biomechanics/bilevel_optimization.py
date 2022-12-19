"""
Implements Werling et al., “Rapid Bilevel Optimization to Concurrently Solve Musculoskeletal Scaling, 
Marker Registration, and Inverse Kinematic Problems for Human Motion Reconstruction.”

Code largely adopted from
https://github.com/keenon/AddBiomechanics/blob/main/server/engine/engine.py
"""

import os
import shutil
import subprocess
import numpy as np
from typing import List, Tuple, Dict
import nimblephysics as nimble


def fit_markers(
    markers: List[List[Dict]],
    model_name="Rajagopal2015_Halpe",
    sex="male",
    heightM=1.7,
    massKg=60,
):  # -> Tuple(List[nimble.biomechanics.MarkerInitialization], nimble.dynamics.Skeleton):

    # get path to this file
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NimbleModels")
    model_file = os.path.join(model_path, model_name + ".osim")

    customOsim = nimble.biomechanics.OpenSimParser.parseOsim(model_file)

    fitter = nimble.biomechanics.MarkerFitter(customOsim.skeleton, customOsim.markersMap)
    fitter.setTriadsToTracking()
    fitter.setInitialIKSatisfactoryLoss(0.05)
    fitter.setInitialIKMaxRestarts(50)
    fitter.setIterationLimit(1000)

    # Create an anthropometric prior
    anthropometrics: nimble.biomechanics.Anthropometrics = nimble.biomechanics.Anthropometrics.loadFromFile(
        os.path.join(model_path, "ANSUR_metrics.xml")
    )

    cols = anthropometrics.getMetricNames()
    cols.append("Heightin")
    cols.append("Weightlbs")
    if sex == "male":
        gauss: nimble.math.MultivariateGaussian = nimble.math.MultivariateGaussian.loadFromCSV(
            os.path.join(model_path, "ANSUR_II_MALE_Public.csv"), cols, 0.001
        )  # mm -> m
    elif sex == "female":
        gauss: nimble.math.MultivariateGaussian = nimble.math.MultivariateGaussian.loadFromCSV(
            os.path.join(model_path, "ANSUR_II_FEMALE_Public.csv"), cols, 0.001
        )  # mm -> m
    else:
        gauss: nimble.math.MultivariateGaussian = nimble.math.MultivariateGaussian.loadFromCSV(
            os.path.join(model_path, "ANSUR_II_BOTH_Public.csv"), cols, 0.001
        )  # mm -> m
    observedValues = {
        "Heightin": heightM * 39.37 * 0.001,
        "Weightlbs": massKg * 2.204 * 0.001,
    }
    gauss = gauss.condition(observedValues)
    anthropometrics.setDistribution(gauss)
    fitter.setAnthropometricPrior(anthropometrics, 0.1)

    results = fitter.runMultiTrialKinematicsPipeline(
        markers,
        nimble.biomechanics.InitialMarkerFitParams()
        .setMaxTrialsToUseForMultiTrialScaling(10)
        .setMaxTimestepsToUseForMultiTrialScaling(4000),
        150,
    )

    return results, customOsim.skeleton


def fetch_formatted_markers(key, augmenter=False):
    from pose_pipeline import TopDownPerson, TopDownMethodLookup
    from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction
    from multi_camera.analysis.biomechanics.opensim import normalize_marker_names

    if augmenter:
        from multi_camera.analysis.biomechanics import opencap_augmenter

        markers, marker_names = opencap_augmenter.convert_markers(key)

        def map_frame(kp3d):
            return {j: k for j, k in zip(marker_names, kp3d)}

        kp3d = [map_frame(k) for k in markers]
        return kp3d

    else:
        method_name = (TopDownMethodLookup & key).fetch1("top_down_method_name")
        joint_names = TopDownPerson.joint_names(method_name)
        joint_names = normalize_marker_names(joint_names)

        kp3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")
        kp3d = kp3d / 1000.0  # mm -> m

        def map_frame(kp3d):
            return {j: k[[1, 2, 0]] for j, k in zip(joint_names, kp3d)}

        return [map_frame(k) for k in kp3d]


def get_trial_performance(
    result: nimble.biomechanics.MarkerInitialization, markers: List[Dict], skeleton: nimble.dynamics.Skeleton
):
    """Returns a dictionary of performance metrics for a single trial

    Args:
        result (nimble.biomechanics.MarkerInitialization): The result of a single trial
        markers (List[Dict]): The markers for a single trial
        skeleton (nimble.dynamics.Skeleton): The skeleton
    Returns:
        Dict: A dictionary of performance metrics
    """

    resultIK = nimble.biomechanics.IKErrorReport(skeleton, result.updatedMarkerMap, result.poses, markers)
    return {
        "averageRootMeanSquaredError": resultIK.averageRootMeanSquaredError,
        "averageMaxError": resultIK.averageMaxError,
        "getSortedMarkerRMSE": resultIK.getSortedMarkerRMSE(),
    }


def save_model(
    model_name: str,
    skeleton_definition: Dict,
    output_path: str,
    mass_kg: float = 60,
    height_m: float = 1.7,
):

    if not os.path.exists(os.path.join(output_path, "Models")):
        os.mkdir(os.path.join(output_path, "Models"))

    # get path to this file
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NimbleModels")
    model_file = os.path.join(model_path, model_name + ".osim")

    # Update custom skeleton
    skeleton = reload_skeleton(model_name, skeleton_definition["group_scales"], skeleton_definition["marker_offsets"])
    fitMarkers = skeleton_definition["marker_offsets"]

    bodyScalesMap: Dict[str, np.ndarray] = {}
    for i in range(skeleton.getNumBodyNodes()):
        bodyNode: nimble.dynamics.BodyNode = skeleton.getBodyNode(i)
        # Now that we adjust the markers BEFORE we rescale the body, we don't want to rescale the marker locations at all
        bodyScalesMap[bodyNode.getName()] = [
            1.0 / bodyNode.getScale()[0],
            1.0 / bodyNode.getScale()[1],
            1.0 / bodyNode.getScale()[2],
        ]

    markerOffsetsMap: Dict[str, Tuple[str, np.ndarray]] = {}
    for n, k in zip(skeleton.getBodyNodes(), fitMarkers):
        markerOffsetsMap[k] = (n.getName(), fitMarkers[k])

    nimble.biomechanics.OpenSimParser.moveOsimMarkers(
        model_file,
        skeleton_definition["body_scale_map"],
        skeleton_definition["marker_offsets_map"],
        os.path.join(output_path, "Models", "unscaled_but_with_optimized_markers.osim"),
    )

    print(output_path)

    nimble.biomechanics.OpenSimParser.saveOsimScalingXMLFile(
        "optimized_scale_and_markers",
        skeleton,
        mass_kg,
        height_m,
        os.path.join(output_path, "Models", "unscaled_but_with_optimized_markers.osim"),
        "Unassigned",
        os.path.join(output_path, "Models", "optimized_scale_and_markers.osim"),
        os.path.join(output_path, "Models", "rescaling_setup.xml"),
    )

    # REQUIRES INSTALLING OpenSim on server
    CMD = "LD_LIBRARY_PATH=/home/jcotton/projects/pose/opensim_dependencies_install/adol-c/lib64:/home/jcotton/projects/pose/opensim_dependencies_install/ipopt/lib/ /home/jcotton/projects/pose/opensim-core-install/bin/opensim-cmd"

    # 8.2.3. Call the OpenSim scaling tool
    # command = "cd " + os.path.join(output_path, "Models") + f" && {CMD} run-tool " + "rescaling_setup.xml"
    command = f"{CMD} run-tool " + os.path.join(output_path, "Models", "rescaling_setup.xml")
    print("Scaling OpenSim files: " + command, flush=True)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()


# kp, skeleton, kp, timestamps, trial, output_dir
def save_trial(
    poses: np.array,
    skeleton: nimble.dynamics.Skeleton,
    markers: List[Dict],
    timestamps: List[float],
    trial_name: str,
    output_path: str,
):
    """Saves a single trial to disk

    Args:
        result (nimble.biomechanics.MarkerInitialization): The result of a single trial
        skeleton (nimble.dynamics.Skeleton): The skeleton
        markers (List[Dict]): The markers for a single trial
        timestamps (List[float]): The timestamps for a single trial
        trial_name (str): The name of the trial
        output_path (str): The path to save the trial to
    """

    print(trial_name)
    if not os.path.exists(os.path.join(output_path, "IK")):
        os.mkdir(os.path.join(output_path, "IK"))
    if not os.path.exists(os.path.join(output_path, "MarkerData")):
        os.mkdir(os.path.join(output_path, "MarkerData"))

    # Write out the .mot files
    nimble.biomechanics.OpenSimParser.saveMot(
        skeleton,
        os.path.join(output_path, "IK/" + trial_name + "_ik.mot"),
        timestamps,
        poses.T,
    )

    nimble.biomechanics.OpenSimParser.saveTRC(
        os.path.join(output_path, "MarkerData/" + trial_name + ".trc"),
        timestamps,
        markers,
    )


def reload_skeleton(model_name: str, body_scales_map, mass_kg=60, height_m=1.7):
    """Reloads a skeleton from a model file

    Args:
        model_name (str): The name of the model
        body_scales_map (Dict[str, np.ndarray]): The body scales map
        marker_offset_map (Dict[str, Tuple[str, np.ndarray]]): The marker offset map
        mass_kg (float): The mass of the model
        height_m (float): The height of the model

    Returns:
        nimble.dynamics.Skeleton: The skeleton
    """

    # get path to this file
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NimbleModels")
    model_file = os.path.join(model_path, model_name + ".osim")

    # Update custom skeleton
    skeleton = nimble.biomechanics.OpenSimParser.parseOsim(model_file).skeleton
    skeleton.setGroupScales(body_scales_map)

    return skeleton


def process_reconstruction_keys(keys: List[dict], output_path: str):
    """Covert 3D reconstructions into OpenSim results

    Args:
        keys: List of keys to process. Should match PersonKeypointReconstruction
    """

    kps = [fetch_formatted_markers(k) for k in keys]
    results, skeleton = bilevel_optimization(kps)

    for result, kp in zip(results, kps):
        print(get_trial_performance(result, kp, skeleton))


if __name__ == "__main__":
    # e.g. python multi_camera/analysis/biomechanics/bilevel_optimization.py --filenames p309_gaitrite_20221024_143143 p309_gaitrite_20221024_143323 video_base_filename --output results/p309
    import argparse
    from multi_camera.datajoint.multi_camera_dj import *
    from multi_camera.analysis.biomechanics import bilevel_optimization

    parser = argparse.ArgumentParser(description="Output OpenSim files from 3D reconstructions")
    parser.add_argument("filenames", nargs="+", help="List of filenames to process")
    parser.add_argument("-t", "--top_down", type=int, default=2, help="Top down method to use")
    parser.add_argument("-r", "--reconstruction_method", type=int, default=2, help="Reconstruction method to use")
    parser.add_argument("-o", "--output", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()
    print(args)
    print([fn for fn in args.filenames])
    keys = [
        (
            PersonKeypointReconstruction * MultiCameraRecording
            & f'video_base_filename LIKE "{fn}" and reconstruction_method={args.reconstruction_method} and top_down_method={args.top_down}'
        ).fetch1("KEY")
        for fn in args.filenames
    ]
    print(keys)
    bilevel_optimization.process_reconstruction_keys(keys, args.output)
