import * as THREE from 'three'
import React, { useEffect, useRef, useContext } from 'react';
import { Viewer } from './visualization_js/viewer.js';
import { AcquisitionState, useEffectOnce } from "../AcquisitionApi";


const system = {
    'meshes': {},
    'geoms': {
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
                'vert_anim': [[
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]],
                'link_idx': 0,
                'transform': {
                    'pos': [0, 0, 1],
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

    const { fetchKeypoints, fetchMesh } = useContext(AcquisitionState);

    useEffectOnce(async () => {
        // You can load data here and update the state accordingly
        system.keypoints = await fetchKeypoints();

        console.log('fetching mesh');
        const mesh = await fetchMesh();

        system.geoms['mesh'][0].vert_anim = mesh.verts;
        system.geoms['mesh'][0].vert = mesh.verts[5];
        system.geoms['mesh'][0].face = mesh.faces;

        console.log("keypoints shape: ", system.keypoints.length, system.keypoints[0].length);

        const domElement = containerRef.current;
        viewerRef.current = new Viewer(domElement, system);
    }, [system]);

    // Set the height of the brax-viewer div to 400 pixels
    const viewerStyle = {
        height: '600px',
        width: '1200px'
    };

    return <div ref={containerRef} id="brax-viewer" style={viewerStyle}></div>;

};

export default BiomechanicalReconstruction;