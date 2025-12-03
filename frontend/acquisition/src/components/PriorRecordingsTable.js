import React, { useContext } from 'react';
import { Table, Accordion, Form } from 'react-bootstrap';
import path from 'path-browserify';
import { AcquisitionState } from "../AcquisitionApi";

const stripPath = (filePath) => path.basename(filePath);

const PriorRecordingsTable = ({ api }) => {

    const { priorRecordings, changeComment, toggleProcess } = useContext(AcquisitionState);

    const handleKeyPress = async (event, participant, filename, newComment) => {
        if (event.key === "Enter") {
            event.preventDefault();
            event.target.blur();
            await changeComment(participant, filename, newComment);
        }
    };

    const sortedRecordings = priorRecordings.sort((a, b) => new Date(a.recording_timestamp) - new Date(b.recording_timestamp)).reverse();

    const formatTimestamp = (timestamp) => {
        const date = new Date(timestamp);
        return new Intl.DateTimeFormat('en', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(date).replace(',', '');
    };

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
                                <th>Timestamp</th>
                                <th>Comment</th>
                                {/* <th>Config</th> */}
                                <th>Timestamp Spread</th>
                                <th>Process</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedRecordings.map((recording) => (
                                <tr key={recording.filename}>
                                    <td>{recording.participant}</td>
                                    <td>{stripPath(recording.filename)}</td>
                                    <td>{formatTimestamp(recording.recording_timestamp)}</td>
                                    <td
                                        contentEditable
                                        suppressContentEditableWarning
                                        // onBlur={(e) => hangeComment(recording.participant, recording.filename, e.target.textContent)}
                                        onKeyDown={(e) => handleKeyPress(e, recording.participant, recording.filename, e.target.textContent)}
                                    >{recording.comment}</td>
                                    {/* <td>{recording.config_file}</td> */}
                                    {/* round recording.timestamp_spread to 2 decimal places */}
                                    <td>{recording.timestamp_spread.toFixed(2)}</td>
                                    <td>

                                        {/* <ToggleButton
                                            id="toggle-check"
                                            type="checkbox"
                                            variant="secondary"
                                            checked={recording.should_process}
                                            value="1"
                                            size="sm"
                                            onChange={(e) => toggleProcess(recording.participant, recording.filename, e.target.checked)}
                                        >
                                            {recording.should_process ? "Yes" : "No"}
                                        </ToggleButton> */}

                                        <Form>
                                            <Form.Check
                                                type="switch"
                                                id={`toggle-check-${recording.filename}`}
                                                // label={recording.should_process ? "Yes" : "No"}
                                                checked={recording.should_process}
                                                onChange={(e) => toggleProcess(recording.participant, recording.filename, e.target.checked)}
                                            />
                                        </Form>
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