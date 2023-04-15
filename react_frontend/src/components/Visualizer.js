import React, { useState, useRef, useContext } from 'react';
import { Container, Row, Button, ToggleButton } from "react-bootstrap";
import { Viewer } from './visualization_js/viewer.js';
import { AcquisitionState, useEffectOnce } from "../AcquisitionApi";

const system = {
    'meshes': {},
    'geoms': {
    },
    'keypoints': null,
    'states': {},
    'dt': 0.033
}

const BiomechanicalReconstruction = ({ data }) => {

    const viewerRef = useRef();
    const containerRef = useRef();
    const guiRef = useRef();

    const [filter, setFilter] = useState(false);

    const { meshUrl, fetchKeypoints, fetchMesh, fetchBiomechanics } = useContext(AcquisitionState);

    // variable to store the websocket connection
    const ws = useRef(null);

    // set showMesh true if data is not none
    const showMesh = { data }.data == "true";

    // Button callbacks
    const onFilterToggle = (val) => {
        console.log("Filter: ", val)
        setFilter(val);
        viewerRef.current.setFilter(val);
    };

    useEffectOnce(async () => {

        console.log('Initializing viewer');


        // You can load data here and update the state accordingly
        //system.keypoints = await fetchKeypoints();
        //console.log("keypoints shape: ", system.keypoints.length, system.keypoints[0].length);

        var faces = null;

        const domElement = containerRef.current;
        const guiElement = guiRef.current;
        viewerRef.current = new Viewer(domElement, system, guiElement);



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
        width: '98%', // magic percentage to match the padding from React
        left: '0px',
        top: '0px',
    };

    const guiStyle = {
        position: 'relative',
        bottom: '0px',
        left: '0%',
        top: '100%',
        width: '100%',
        height: '0px',
        zIndex: '1000'
    }

    return (
        <Container>
            <Row className='p-2'>
                <div ref={guiRef} id="gui" style={guiStyle}></div>
                <div ref={containerRef} id="brax-viewer" style={viewerStyle}></div>
            </Row>

            <Row className='p-4' sm={4}>
                <ToggleButton className='mb-2'
                    id="toggle-check"
                    type="checkbox"
                    variant="outline-primary"
                    value="1"
                    checked={filter}
                    onChange={(e) => onFilterToggle(e.currentTarget.checked)} >
                    Filter
                </ToggleButton>
                {/* add space between buttons */}
                <div className='p-2'></div>
                <Button className='p-2'>Annotate</Button>
            </Row>
        </Container>
    );

};

export default BiomechanicalReconstruction;