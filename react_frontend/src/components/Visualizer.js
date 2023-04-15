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

    const { meshUrl, fetchKeypoints, fetchMesh, fetchBiomechanics } = useContext(AcquisitionState);

    const ws = useRef(null);

    // set showMesh true if data is not none
    const showMesh = { data }.data == "true";

    useEffectOnce(async () => {

        console.log('Initializing viewer');


        // You can load data here and update the state accordingly
        system.keypoints = await fetchKeypoints();
        console.log("keypoints shape: ", system.keypoints.length, system.keypoints[0].length);

        var faces = null;

        const domElement = containerRef.current;
        viewerRef.current = new Viewer(domElement, system);


        if (showMesh) {
            console.log('Connecting to mesh websocket...');

            ws.current = new WebSocket(meshUrl);

            ws.current.onopen = () => {
                console.log("Mesh WebSocket connected");
            };

            ws.current.onmessage = (event) => {
                // convert event.data from JSON to JS object
                const data = JSON.parse(event.data);

                if (faces === null) {
                    // unpack as json for first message:
                    faces = data.faces;
                } else {
                    viewerRef.current.addFrame(data, faces);
                }

            };

            ws.current.onclose = () => {
                console.log("Mesh WebSocket disconnected");
            };

            ws.current.onerror = (event) => {
                console.log("Mesh WebSocket error observed:", event);
            }

            return () => {
                if (ws.current) {
                    ws.current.close();
                }
            };
        } else {
            const biomechanics = await fetchBiomechanics();
            viewerRef.current.addBiomechanics(biomechanics.meshes, biomechanics.trajectories);
            console.log("biomechanics: ", biomechanics);

            return () => {
                console.log("Closed biomechanics")
            };
        }

    }, []);

    // Set the height of the brax-viewer div to 400 pixels
    const viewerStyle = {
        height: '900px',
        width: '100%'
    };

    return <div ref={containerRef} id="brax-viewer" style={viewerStyle}></div>;

};

export default BiomechanicalReconstruction;