import { useContext, useState } from "react";
import { Container, Row, Col, Form, Button } from "react-bootstrap";
import { AcquisitionState } from "../AcquistionApi";
import Spinner from 'react-bootstrap/Spinner';

const RecordingControl = () => {
    const { newTrial, recordingFilename, recordingSystemStatus, calibrationVideo, previewVideo, stopAcquisition } = useContext(AcquisitionState);

    const [comment, setComment] = useState("");

    return (
        <div >
            <Form className="p-2 border">

                <Container>
                    <Row className="justify-content-md-left">
                        <Col md="auto">
                            <Button id="preview" className="btn btn-secondary"
                                onClick={() => previewVideo()}
                            >Preview</Button>
                        </Col>
                        <Col md="auto">
                            <Button id="calibration" className="btn btn-secondary"
                                onClick={() => calibrationVideo()}
                            >Calibration</Button>
                        </Col>
                        <Col md="auto">
                            <Button id="new_trial" onClick={() => newTrial(comment)}
                                className="btn btn-primary">New Trial</Button>
                        </Col>
                        <Col md="auto">
                            <Button id="stop" className="btn btn-danger float-right"
                                onClick={() => stopAcquisition()}
                            >Stop</Button>
                        </Col>
                        <Col md="auto">
                            {recordingSystemStatus == "Recording" ? <Spinner animation="border" role="recording" /> : null}
                        </Col>
                    </Row>
                </Container>

                <Form.Group as={Row} className="p-2">
                    <Form.Label column sm={3}>Comment:</Form.Label>
                    <Col sm={6}>
                        <Form.Control id="comment" type="text" placeholder="Comment" onChange={(e) => setComment(e.target.value)}
                        />
                    </Col>
                </Form.Group>

                <Form.Group as={Row} className="p-2">
                    <Form.Label column sm={3}>File Name:</Form.Label>
                    <Col sm={6}>
                        <Form.Control id="file_name" type="text" value={recordingFilename} readOnly />
                    </Col>
                </Form.Group>

                {/* Show the spinner when recordingSystemStatus=Recording */}

                <Row className="justify-content-md-left">
                    <Col>
                        <h1> Recording Status: {recordingSystemStatus} </h1>
                    </Col>
                </Row>

            </Form>
        </div >
    );

};

export default RecordingControl;