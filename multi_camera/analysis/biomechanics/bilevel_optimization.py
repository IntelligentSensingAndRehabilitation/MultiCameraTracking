"""
Implements Werling et al., “Rapid Bilevel Optimization to Concurrently Solve Musculoskeletal Scaling, 
Marker Registration, and Inverse Kinematic Problems for Human Motion Reconstruction.”

Code largely adopted from
https://github.com/keenon/AddBiomechanics/blob/main/server/engine/engine.py

It is a little awkward in palces as the system expects to pass data around as files. However
this isn't a major issue since we will mostly want to load these results into OpenSim anyway.

Nimble gets a bit flaky with Jupyter so also worth running this in a script. The default calling
of this script allows filtering by a subject name and date.
"""

import os
import shutil
import subprocess
import numpy as np
from typing import List, Tuple, Dict
import nimblephysics as nimble

# from nimblephysics.loader import absPath


def bilevel_optimization(
    trc_file_path: List[str],
    output_path: str,
    model_name="/home/jcotton/projects/pose/MultiCameraTracking/notebooks/biomechanics_fit/Rajagopal2015_Halpe.osim",
    model_path="/home/jcotton/projects/pose/MultiCameraTracking/notebooks/biomechanics_fit",
    run_opensim=False,
    sex="male",
    heightM=1.7,
    massKg=60,
):

    model_path = os.path.split(model_name)[0]
    print(f"Model name: {model_name}, model path: {model_path}")
    print(f"trc_file_path: {trc_file_path}")
    shutil.copy(model_name, os.path.join(output_path, "fit.osim"))

    nimble.biomechanics.OpenSimParser.rationalizeJoints(
        os.path.join(output_path, "fit.osim"), os.path.join(output_path, "fit_rational.osim")
    )
    customOsim = nimble.biomechanics.OpenSimParser.parseOsim(os.path.join(output_path, "fit_rational.osim"))

    customOsim.skeleton.autogroupSymmetricSuffixes()
    if customOsim.skeleton.getBodyNode("hand_r") is not None:
        customOsim.skeleton.setScaleGroupUniformScaling(customOsim.skeleton.getBodyNode("hand_r"))
    customOsim.skeleton.autogroupSymmetricPrefixes("ulna", "radius")

    fitter = nimble.biomechanics.MarkerFitter(customOsim.skeleton, customOsim.markersMap)
    fitter.setInitialIKSatisfactoryLoss(0.05)
    # fitter.setInitialIKSatisfactoryLoss(0.5)
    fitter.setInitialIKMaxRestarts(50)
    # fitter.setIterationLimit(500)
    fitter.setIterationLimit(500)
    fitter.setTrackingMarkers(customOsim.trackingMarkers)

    # This is 1.0x the values in the default code
    fitter.setRegularizeAnatomicalMarkerOffsets(1.0)
    # This is 1.0x the default value
    fitter.setRegularizeTrackingMarkerOffsets(0.05)
    # These are 2x the values in the default code
    # fitter.setMinSphereFitScore(3e-5 * 2)
    # fitter.setMinAxisFitScore(6e-5 * 2)
    fitter.setMinSphereFitScore(0.01)
    fitter.setMinAxisFitScore(0.001)
    # Default max joint weight is 0.5, so this is 2x the default value
    fitter.setMaxJointWeight(1.0)

    trcFiles = []
    for trc_file in trc_file_path:
        trcFile = nimble.biomechanics.OpenSimParser.loadTRC(trc_file)
        trialErrorReport = fitter.generateDataErrorsReport(trcFile.markerTimesteps)
        print(fitter.checkForEnoughMarkers(trcFile.markerTimesteps), trialErrorReport)
        trcFiles.append(trcFile)

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
    fitter.setAnthropometricPrior(anthropometrics, 0.001)

    results: List[nimble.biomechanics.MarkerInitialization] = fitter.runMultiTrialKinematicsPipeline(
        [t.markerTimesteps for t in trcFiles],
        nimble.biomechanics.InitialMarkerFitParams()
        .setMaxTrialsToUseForMultiTrialScaling(5)
        .setMaxTimestepsToUseForMultiTrialScaling(4000),
        150,
    )

    # Update custom skeleton
    customOsim.skeleton.setGroupScales(results[0].groupScales)
    fitMarkers: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]] = results[0].updatedMarkerMap

    # 8.2. Write out the usable OpenSim results
    if not os.path.exists(os.path.join(output_path, "results")):
        os.mkdir(os.path.join(output_path, "results"))
    if not os.path.exists(os.path.join(output_path, "results", "IK")):
        os.mkdir(os.path.join(output_path, "results", "IK"))
    if not os.path.exists(os.path.join(output_path, "results", "ID")):
        os.mkdir(os.path.join(output_path, "results", "ID"))
    if not os.path.exists(os.path.join(output_path, "results", "C3D")):
        os.mkdir(os.path.join(output_path, "results", "C3D"))
    if not os.path.exists(os.path.join(output_path, "results", "Models")):
        os.mkdir(os.path.join(output_path, "results", "Models"))
    # if exportMJCF and not os.path.exists(path+'results/MuJoCo'):
    #    os.mkdir(path+'results/MuJoCo')
    # if exportSDF and not os.path.exists(path+'results/SDF'):
    #    os.mkdir(path+'results/SDF')
    if not os.path.exists(os.path.join(output_path, "results", "MarkerData")):
        os.mkdir(os.path.join(output_path, "results", "MarkerData"))

    # 8.2.1. Adjusting marker locations
    print("Adjusting marker locations on scaled OpenSim file", flush=True)
    bodyScalesMap: Dict[str, np.ndarray] = {}
    for i in range(customOsim.skeleton.getNumBodyNodes()):
        bodyNode: nimble.dynamics.BodyNode = customOsim.skeleton.getBodyNode(i)
        # Now that we adjust the markers BEFORE we rescale the body, we don't want to rescale the marker locations at all
        bodyScalesMap[bodyNode.getName()] = [1, 1, 1]  # bodyNode.getScale()
    markerOffsetsMap: Dict[str, Tuple[str, np.ndarray]] = {}
    markerNames: List[str] = []
    for k in fitMarkers:
        v = fitMarkers[k]
        markerOffsetsMap[k] = (v[0].getName(), v[1])
        markerNames.append(k)
    nimble.biomechanics.OpenSimParser.moveOsimMarkers(
        os.path.join(output_path, "fit_rational.osim"),
        bodyScalesMap,
        markerOffsetsMap,
        os.path.join(output_path, "results", "Models", "unscaled_but_with_optimized_markers.osim"),
    )

    print(markerOffsetsMap)

    for trc_file, result in zip(trc_file_path, results):
        trialName = os.path.split(trc_file)[1].split(".")[0]

        trialProcessingResult = {}

        resultIK = nimble.biomechanics.IKErrorReport(
            customOsim.skeleton, fitMarkers, result.poses, trcFile.markerTimesteps
        )
        trialProcessingResult["autoAvgRMSE"] = resultIK.averageRootMeanSquaredError
        trialProcessingResult["autoAvgMax"] = resultIK.averageMaxError
        trialProcessingResult["markerErrors"] = resultIK.getSortedMarkerRMSE()

        print(trialProcessingResult)

        # Write out the .mot files
        nimble.biomechanics.OpenSimParser.saveMot(
            customOsim.skeleton,
            os.path.join(output_path, "results/IK/" + trialName + "_ik.mot"),
            trcFile.timestamps,
            result.poses,
        )
        resultIK.saveCSVMarkerErrorReport(
            os.path.join(output_path, "results/IK/" + trialName + "_ik_per_marker_error_report.csv")
        )
        # nimble.biomechanics.OpenSimParser.saveGRFMot(
        #    os.path.join(path, 'results/ID/'+trialName+'_grf.mot'), trcFile.timestamps, forcePlates)
        nimble.biomechanics.OpenSimParser.saveTRC(
            os.path.join(output_path, "results/MarkerData/" + trialName + ".trc"),
            trcFile.timestamps,
            trcFile.markerTimesteps,
        )
        # if c3dFile is not None:
        #    shutil.copyfile(trialPath + 'markers.c3d', path +
        #                    'results/C3D/' + trialName + '.c3d')

    # 8.2.2. Write the XML instructions for the OpenSim scaling tool
    nimble.biomechanics.OpenSimParser.saveOsimScalingXMLFile(
        "optimized_scale_and_markers",
        customOsim.skeleton,
        massKg,
        heightM,
        "Models/unscaled_but_with_optimized_markers.osim",
        "Unassigned",
        "Models/optimized_scale_and_markers.osim",
        os.path.join(output_path, "results", "Models", "rescaling_setup.xml"),
    )

    if run_opensim:
        # REQUIRES INSTALLING OpenSim on server
        CMD = "LD_LIBRARY_PATH=/home/jcotton/projects/pose/opensim_dependencies_install/adol-c/lib64:/home/jcotton/projects/pose/opensim_dependencies_install/ipopt/lib/ /home/jcotton/projects/pose/opensim-core-install/bin/opensim-cmd"
        # 8.2.3. Call the OpenSim scaling tool
        command = (
            "cd "
            + os.path.join(output_path, "results")
            + f" && {CMD} run-tool "
            + os.path.join(output_path, "results/Models/rescaling_setup.xml")
        )
        print("Scaling OpenSim files: " + command, flush=True)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()


def process_reconstruction_keys(keys: List[dict]):
    """Covert 3D reconstructions into OpenSim results

    Args:
        keys: List of keys to process. Should match PersonKeypointReconstruction
    """
    from ...datajoint.multi_camera_dj import (
        MultiCameraRecording,
        PersonKeypointReconstruction,
        PersonKeypointReconstructionMethodLookup,
    )
    from pose_pipeline import TopDownMethodLookup
    import tempfile

    reconstruction_method_name = np.unique(
        (PersonKeypointReconstructionMethodLookup & keys).fetch("reconstruction_method_name")
    )
    assert len(reconstruction_method_name) == 1, "Multiple reconstruction methods found"

    method_name = np.unique((TopDownMethodLookup & keys).fetch("top_down_method_name"))
    assert len(method_name) == 1, "Multiple keypoint types found"

    trc_files = []
    if method_name == "MMPoseHalpe":
        with tempfile.TemporaryDirectory() as temp_dir:

            for key in keys:
                fn = (MultiCameraRecording & key).fetch1("video_base_filename")
                file_path = os.path.join(temp_dir, fn + ".trc")
                (PersonKeypointReconstruction & key).export_trc(file_path)
                trc_files.append(file_path)

            model_file = (
                "/home/jcotton/projects/pose/MultiCameraTracking/notebooks/biomechanics_fit/Rajagopal2015_Halpe.osim"
            )
            output_path = "/home/jcotton/projects/pose/MultiCameraTracking/notebooks/biomechanics_fit/results2/"

            bilevel_optimization(trc_files, output_path, model_file, run_opensim=True)


if __name__ == "__main__":
    from multi_camera.datajoint.multi_camera_dj import *
    from multi_camera.analysis.biomechanics import bilevel_optimization

    keys = (
        PersonKeypointReconstruction * MultiCameraRecording
        & '(video_base_filename LIKE "p309_gaitrite_20221024_143143" or video_base_filename LIKE "p309_gaitrite_20221024_143323" or video_base_filename LIKE "p309_gaitrite_20221024_143627")and reconstruction_method=2'
        # & 'video_base_filename LIKE "p310%" and reconstruction_method=0'
    ).fetch("KEY")
    bilevel_optimization.process_reconstruction_keys(keys)
