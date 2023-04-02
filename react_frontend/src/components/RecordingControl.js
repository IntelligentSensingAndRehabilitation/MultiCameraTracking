import React from 'react';
import { useContext, useState } from "react";
import { Container, Row, Col, Form, Button } from "react-bootstrap";
import { AcquisitionState } from "../AcquisitionApi";
import Spinner from 'react-bootstrap/Spinner';

const RecordingControl = () => {
    const { newTrial, recordingFilename, recordingSystemStatus, calibrationVideo, previewVideo, stopAcquisition } = useContext(AcquisitionState);

    const [comment, setComment] = useState("");
    const [maxFrames, setMaxFrames] = useState(1000);

    return (
        <div >
            <Form className="g-4 p-2 border">

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

            </Form>
        </div >
    );

};

export default RecordingControl;