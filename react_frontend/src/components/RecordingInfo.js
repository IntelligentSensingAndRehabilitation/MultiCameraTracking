import React from 'react';
import ReactDOM from 'react-dom';
import { useEffect, useContext, useRef } from "react";
import { Row, Col, Form } from "react-bootstrap";
import { AcquisitionState } from "../AcquistionApi";

const RecordingInfo = () => {
    const { recordingDir, recordingFileBase } = useContext(AcquisitionState);

    const recordingDirRef = useRef(null);
    const recordingBaseRef = useRef(null);

    useEffect(() => {
        const form1 = ReactDOM.findDOMNode(recordingDirRef.current);
        form1.value = recordingDir;

        const form2 = ReactDOM.findDOMNode(recordingBaseRef.current);
        form2.value = recordingFileBase;
    }, [recordingDir, recordingFileBase]);

    return (
        <div >
            <Form className="p-2">
                <Form.Group controlId="recordingDirForm" as={Row} className="p-1">
                    <Form.Label column sm={3}>Recording Dir: </Form.Label>
                    <Col sm={6}>
                        <Form.Control
                            ref={recordingDirRef}
                            type="text"
                            placeholder="Recording Dir (Autosets)"
                            defaultValue={recordingDir}
                        >
                        </Form.Control>
                    </Col>
                </Form.Group>

                <Form.Group controlId="recordingFileBaseForm" as={Row} className="p-1">
                    <Form.Label column sm={3}>Recording Base Filename: </Form.Label>
                    <Col sm={6}>
                        <Form.Control
                            ref={recordingBaseRef}
                            type="text"
                            placeholder="Base Filename"
                            defaultValue={recordingFileBase}
                        >
                        </Form.Control>
                    </Col>
                </Form.Group>
            </Form>
        </div>
    );

};

export default RecordingInfo;