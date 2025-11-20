import React from 'react';
import { useContext, useState } from "react";
import { Row, Col, Form, Button } from "react-bootstrap";
import { AcquisitionState } from "../AcquisitionApi";

const Config = () => {

    const { currentConfig, availableConfigs, resetCameras, updateConfig } = useContext(AcquisitionState);

    const [validated, setValidated] = useState(false);

    const handleConfigChange = async (event) => {
        const value = event.target.value;
        console.log("handleConfigChange: " + value);
        setValidated(false)
        await updateConfig(value);
        setValidated(true)
        console.log("Validated: " + validated)
    };

    return (
        <div >
            <Form noValidate validated={validated} className="p-2">
                <Form.Group controlId="config" as={Row}>
                    <Form.Label column sm={3}>Config:</Form.Label>
                    <Col sm={6}>
                        <Form.Control
                            as="select"
                            value={currentConfig}
                            onChange={handleConfigChange}
                            className="flex-grow-1"
                        >
                            {availableConfigs.map((config) => (
                                <option value={config} key={config}>
                                    {config}
                                </option>
                            ))}
                        </Form.Control>
                    </Col>
                    <Col sm={3} className="d-flex justify-content-end">
                        <Button variant="primary" onClick={() => resetCameras()}>
                            Reset Cameras
                        </Button>
                    </Col>

                </Form.Group>
            </Form>
        </div>
    );

};

export default Config;