import React from 'react';
import { useContext, useState } from "react";
import { Container, Row, Col, Form, Button, ProgressBar } from "react-bootstrap";
import { AcquisitionState } from "../AcquisitionApi";
import Spinner from 'react-bootstrap/Spinner';

const RecordingControl = () => {
    const { newTrial, recordingFilename, recordingProgress, recordingSystemStatus, calibrationVideo, previewVideo, stopAcquisition } = useContext(AcquisitionState);

    const [comment, setComment] = useState("");
    const [maxFrames, setMaxFrames] = useState(1000);

    return (
        <div >

            {/* Set the form css background-color to green while recordingSystemStatus === "Recording"*/}
            <style type="text/css">
                {`
                .form-recording {
                    background-color: purple;
                    color: white; 
                }`
                }

            </style>

            {/* Set the form color green while recordingSystemStatus === "Recording"*/}
            <Form className={recordingSystemStatus === "Recording" ? "g-4 p-2 border bg-warning" : "g-4 p-2 border"}>

                <Container>
                    <Row className="justify-content-md-left">
                        <Col md="auto">
                            <Button id="preview"
                                className="btn btn-secondary"
                                disabled={recordingSystemStatus !== "Idle"}
                                onClick={() => previewVideo(maxFrames)}
                            >Preview</Button>
                        </Col>
                        <Col md="auto">
                            <Button id="calibration"
                                className="btn btn-secondary"
                                disabled={recordingSystemStatus !== "Idle"}
                                onClick={() => calibrationVideo(maxFrames)}
                            >Calibration</Button>
                        </Col>
                        <Col md="auto">
                            <Button id="new_trial"
                                disabled={recordingSystemStatus !== "Idle"}
                                onClick={() => newTrial(comment, maxFrames)}
                                className="btn btn-primary">New Trial</Button>
                        </Col>
                        <Col md="auto">
                            <Button id="stop"
                                className="btn btn-danger float-right"
                                disabled={recordingSystemStatus === "Idle"}
                                onClick={() => stopAcquisition()}
                            >Stop</Button>
                        </Col>
                        <Col md="auto">
                            {recordingSystemStatus === "Recording" ? <Spinner animation="border" role="status" /> : null}
                        </Col>
                    </Row>
                </Container>

                <Form.Group as={Row} controlId="max_frames" className="p-2">
                    <Form.Label column sm={3}>Max Frames:</Form.Label>
                    <Col sm={6}>
                        <Form.Control type="number" value={maxFrames} onChange={(e) => setMaxFrames(e.target.value)} />
                    </Col>
                </Form.Group>

                <Form.Group as={Row} controlId="comment" className="p-2">
                    <Form.Label column sm={3}>Comment:</Form.Label>
                    <Col sm={6}>
                        <Form.Control type="text" placeholder="Comment" onChange={(e) => setComment(e.target.value)} />
                    </Col>
                </Form.Group>

                <Form.Group as={Row} controlId="file_name" className="p-2">
                    <Form.Label column sm={3}>File Name:</Form.Label>
                    <Col sm={6}>
                        <Form.Control type="text" value={recordingFilename} readOnly />
                    </Col>
                </Form.Group>

                <Row >
                    <Col sm={3}>
                        Recording Status:
                    </Col>
                    <Col sm={6} className="text-start">
                        {' '} {recordingSystemStatus}
                    </Col>
                </Row>


                <Row className="p-2">
                    <Col sm={3}>
                        <Form.Label >Recording Progress:</Form.Label>
                    </Col>
                    <Col>
                        {/* Only show progress bar if status is recording */}
                        {recordingSystemStatus !== "Recording" ? null :
                            <ProgressBar now={recordingProgress} label={`${recordingProgress}%`} />
                        }
                    </Col>
                </Row>
            </Form>
        </div >
    );

};

export default RecordingControl;