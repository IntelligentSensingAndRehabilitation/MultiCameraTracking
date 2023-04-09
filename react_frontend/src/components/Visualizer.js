import * as THREE from 'three'
import React, { useEffect, useRef, useContext } from 'react';
import { Viewer } from './visualization_js/viewer.js';
import { AcquisitionState, useEffectOnce } from "../AcquisitionApi";

const HumanMesh = ({ data }) => {
    // Load your human 3D model and update the human mesh with the data
    // ...
    // Use the following line to set the position and rotation:
    // mesh.position.fromArray(data.position);
    // mesh.quaternion.fromArray(data.quaternion);
};

const system = {
    'meshes': {},
    'geoms': {
        // 'sphere': [
        //     {
        //         'name': 'Sphere',
        //         'radius': 0.1,
        //         'transform': {
        //             'pos': [10, 10, 1],
        //         },
        //         'rgba': [0, 0, 1, 1],
        //     },
        // ],
        'mesh': [
            {
                'name': 'Mesh',
                'vert': [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                'face': [
                    [0, 1, 2],
                    [0, 1, 3],
                    [0, 2, 3],
                    [1, 2, 3],
                ],
                'link_idx': 0,
                'transform': {
                    'pos': [2, 2, 2],
                },
                'rgba': [1, 0, 0, 1],
            },
        ],
    },
    'keypoints': null,
    'states': { 'x': { 'pos': [] } }, 'dt': 0.033
}

const BiomechanicalReconstruction = ({ data }) => {

    const viewerRef = useRef();
    const containerRef = useRef();

    const { keypoints, fetchKeypoints, fetchMesh } = useContext(AcquisitionState);

    // useEffectOnce(() => {
    //     // Instantiate the Viewer when the component mounts
    //     const domElement = containerRef.current;
    //     viewerRef.current = new Viewer(domElement, system);
    // }, [system]);

    useEffectOnce(async () => {
        // You can load data here and update the state accordingly
        system.keypoints = await fetchKeypoints();

        console.log('fetching mesh');
        const mesh = await fetchMesh();
        console.log('mesh: ' + mesh + " keypoints: " + system.keypoints);
        system.geoms['mesh'][0].vert = mesh.verts[0];
        system.geoms['mesh'][0].face = mesh.faces;
        //console.log('received mesh. size: ', mesh.vert.length, mesh.face.length);
        console.log(system.geoms)
        console.log(mesh.verts)

        console.log("keypoints shape: ", system.keypoints.length, system.keypoints[0].length);
        const domElement = containerRef.current;
        console.log("domElement: ", domElement)
        viewerRef.current = new Viewer(domElement, system);
    }, [system]);

    // Set the height of the brax-viewer div to 400 pixels
    const viewerStyle = {
        height: '400px',
        width: '800px'
    };

    return <div ref={containerRef} id="brax-viewer" style={viewerStyle}></div>;

};

export default BiomechanicalReconstruction;