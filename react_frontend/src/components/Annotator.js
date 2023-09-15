import React, { useState, useEffect, useRef, useContext } from 'react';
import { Container, Row, Col, Form, Button, ToggleButton } from "react-bootstrap";
import { Viewer } from './visualization_js/viewer.js';
import { AcquisitionState } from "../AcquisitionApi.js";

const system = {
    'meshes': {},
    'geoms': {},
    'smpl': {},
    'keypoints': null,
    'states': {},
    'dt': 0.166 // only showing every 5 frame at 30 fps
}

const Annotator = ({ data }) => {

    const viewerRef = useRef();
    const containerRef = useRef();
    const guiRef = useRef();

    const [filter, setFilter] = useState(false);

    const { fetchMesh, fetchUnannotatedRecordings, annotateRecording } = useContext(AcquisitionState);

    const [currentRecording, setCurrentRecording] = useState(null);
    const [recordingValidated, setRecordingValidated] = useState(false);
    const [unannotatedRecordings, setUnannotatedRecordings] = useState([]);
    const [downsampling, setDownsampling] = useState(5);

    const refreshRecordings = () => {
        fetchUnannotatedRecordings().then((data) => {
            // prepend an empty string to the list
            data.unshift("");
            setUnannotatedRecordings(data);
        });
    }

    useEffect(() => {
        console.log("Fetching unannotated recordings...")
        refreshRecordings();
    }, []);

    // Button callbacks
    const onFilterToggle = (val) => {
        console.log("Filter: ", val)
        setFilter(val);
        viewerRef.current.setFilter(val);
    };

    useEffect(() => {
        console.log("Current recording: ", currentRecording)
        if (currentRecording === "") {
            // TODO: close the viewer data
            setRecordingValidated(true);
        }
        else {
            if (viewerRef.current != null) {
                viewerRef.current.close();
                // delete the viewerRef.current
                viewerRef.current = null;
            }

            setRecordingValidated(false);

            if (currentRecording != null && currentRecording !== "") {
                fetchMesh(currentRecording, downsampling).then((data) => {
                    console.log("Mesh data: ", data)
                    system.smpl = data;

                    const domElement = containerRef.current;
                    const guiElement = guiRef.current;
                    viewerRef.current = new Viewer(domElement, system, guiElement);

                    setRecordingValidated(true);
                });
            }
        }
    }, [currentRecording, downsampling]);

    const onAnnotate = () => {
        if (recordingValidated) {
            console.log("Annotating recording: ", currentRecording)
            const ids = viewerRef.current.getSelectedIds();
            console.log("Selected ids: ", ids)
            if (ids != null && ids.length > 0) {
                annotateRecording(currentRecording, ids).then((data) => {
                    console.log("Annotate response: ", data)
                    if (data) {
                        // close the viewer
                        viewerRef.current.close();
                        // delete the viewerRef.current
                        viewerRef.current = null;
                        // set the current recording to null
                        setCurrentRecording("");
                        // fetch the unannotated recordings again
                        refreshRecordings();
                    }
                });
            }
        }
    }

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

            <Form noValidate validated={recordingValidated} className="p-2">
                <Form.Group controlId="recording" as={Row}>
                    <Form.Label column sm={3}>Unannotated Recordings:</Form.Label>
                    <Col sm={6}>
                        <Form.Control
                            as="select"
                            value={currentRecording}
                            onChange={(e) => setCurrentRecording(e.target.value)}
                            className="flex-grow-1"
                        >
                            {unannotatedRecordings.map((recording) => (
                                <option value={recording} key={recording}>
                                    {recording}
                                </option>
                            ))}
                        </Form.Control>
                    </Col>

                </Form.Group>

                <div className='p-2'></div>

                <Form.Group controlId="downsampling" as={Row} >
                    <Form.Label column sm={3}>Downsampling</Form.Label>
                    <Col sm={6}>
                        <Form.Control as="select" value={downsampling} onChange={(e) => setDownsampling(e.target.value)}>
                            <option>1</option>
                            <option>2</option>
                            <option>5</option>
                            <option>10</option>
                            <option>15</option>
                        </Form.Control>
                    </Col>
                </Form.Group>
            </Form>


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
                <Button className='p-2'
                    variant="primary"
                    onClick={onAnnotate}>
                    Annotate</Button>
            </Row>
        </Container>
    );

};

export default Annotator;