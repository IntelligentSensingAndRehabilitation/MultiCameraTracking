import React, { useContext } from 'react';
import { Table, ToggleButton, Accordion } from 'react-bootstrap';
import { AcquisitionState } from "../AcquisitionApi";

const PriorRecordingsTable = ({ api }) => {

    const { priorRecordings } = useContext(AcquisitionState);

    return (
        <Accordion defaultActiveKey="0" className="g-4 p-2" >
            <Accordion.Item eventKey="0">
                <Accordion.Header>Recordings</Accordion.Header>
                <Accordion.Body>

                    <Table id="prior_recordings_table" striped bordered hover>
                        <thead>
                            <tr>
                                <th>Participant</th>
                                <th>Filename</th>
                                <th>Comment</th>
                                {/* <th>Config</th> */}
                                <th>Process</th>
                            </tr>
                        </thead>
                        <tbody>
                            {priorRecordings.map((recording) => (
                                <tr key={recording.filename}>
                                    <td>{recording.participant}</td>
                                    <td>{recording.filename}</td>
                                    <td>{recording.comment}</td>
                                    {/* <td>{recording.config_file}</td> */}
                                    <td>

                                        <ToggleButton
                                            id="toggle-check"
                                            type="checkbox"
                                            variant="secondary"
                                            checked={recording.should_process}
                                            value="1"
                                            size="sm"
                                            disabled
                                        // onChange={(e) => setChecked(e.currentTarget.checked)}
                                        >
                                            {recording.should_process ? "Yes" : "No"}
                                        </ToggleButton>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </Table>
                </Accordion.Body>
            </Accordion.Item>
        </Accordion >
    );
};

export default PriorRecordingsTable;