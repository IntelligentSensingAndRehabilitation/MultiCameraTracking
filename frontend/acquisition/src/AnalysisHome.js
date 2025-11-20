import React, { useContext, useState, useRef, useEffect } from "react";
import { Accordion, Table, Form, Container, Button, Col, Row, ButtonGroup, ToggleButton } from 'react-bootstrap';
import path from 'path-browserify';
import { AcquisitionState } from "./AcquisitionApi";

const stripPath = (filePath) => path.basename(filePath);


const RecordingTable = ({ recordings, participant, session_date, imported, callback }) => {

    const videoProjectRef = useRef(null);

    const handleToggle = async (recordingKey, isChecked) => {
        await callback.onToggleShouldProcess(participant, session_date, recordingKey, isChecked);
    };

    const calibrate = async (recordingKey, isCharuco) => {
        await callback.onCalibrate(participant, session_date, recordingKey, isCharuco);
    }

    const process = async () => {
        const videoProject = videoProjectRef.current.value;
        await callback.onProcessSession(participant, session_date, videoProject);
    }

    return (
        <Container>
            <Table striped hover borderless responsive>
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th>Comment</th>
                        {/* <th>Config File</th> */}
                        <th>Should Process</th>
                    </tr>
                </thead>
                <tbody>
                    {recordings.map((recording, index) => (
                        <tr key={recording.filename}>
                            <td>{stripPath(recording.filename)}</td>


                            {/* If recording.comment == "calibration" add button for run calibration, otherwise show comment */}
                            {recording.comment === "calibration" ? (
                                <td><Button variant="primary" onClick={() => calibrate(recording.filename, false)}>Checkerboard Calibration</Button></td>
                            ) : recording.comment === "charuco" ? (
                                <td><Button variant="primary" onClick={() => calibrate(recording.filename, true)}>Charuco Calibration</Button></td>
                            ) : (
                                <td>{recording.comment}</td>
                            )}


                            {/* <td>{recording.config_file}</td> */}
                            <td>
                                {/* {recording.should_process ? 'Yes' : 'No'} */}
                                <Form.Check
                                    type="switch"
                                    id={`toggle-${index}`}
                                    checked={recording.should_process}
                                    onChange={(e) => handleToggle(recording.filename, e.target.checked)}
                                    disabled={imported}
                                />
                                {/* Now have label and form for video project entry */}

                            </td>
                        </tr>
                    ))}
                </tbody>
            </Table>
            {/* Button that pushes a recording to DataJoint */}
            <Form.Group controlId="video_project" className="p-2">
                <Row>
                    <Col>
                        <Form.Label>Video Project</Form.Label>
                    </Col>
                    <Col>
                        <Form.Control type="text" ref={videoProjectRef} placeholder="Enter Project" disabled={imported} />
                    </Col>
                    <Col>
                        <Button variant="primary" onClick={() => process()} disabled={imported}>
                            Push to DataJoint
                        </Button>
                    </Col>
                </Row>
            </Form.Group>

        </Container>
    );
};

const InnerAccordion = ({ sessions, participants, callback, sortByDate }) => {

    console.log("sessions", sessions, "participant", participants)

    if (sortByDate) {
        const date = sessions; // only 1 in this case

        // make a deep copy of the participants
        var participants_copy = JSON.parse(JSON.stringify(participants))

        console.log("participants_copy", participants_copy)

        // need to filter the sessions for each participant to only include the one with the correct date
        participants_copy.forEach(participant => {
            participant.sessions = participant.sessions.filter(session => session.session_date === date)
            if (participant.sessions.length === 0) {
                console.log("participant", participant, "has no sessions for date", date)
            }
        })

        participants_copy = participants_copy.filter(participant => participant.sessions.length > 0)


        return (
            < Accordion defaultActiveKey="0" >
                {
                    participants_copy.map((participant, index) => (
                        <Accordion.Item key={participant.sessions[0].session_path} eventKey={`${index}`}>
                            <Accordion.Header>{participant.name}</Accordion.Header>
                            <Accordion.Body className="p-1" >
                                <RecordingTable recordings={participant.sessions[0].recordings} participant={participant.name} session_date={participant.sessions[0].session_date} imported={participant.sessions[0].imported} callback={callback} />
                            </Accordion.Body>
                        </Accordion.Item>
                    ))
                }
            </Accordion >
        );
    } else {
        // When not sorting by date, there is one participant for the inner accordion
        // and the sessions are all from that person so we loop over them
        const participant = participants; // only 1 in this case
        return (
            < Accordion defaultActiveKey="0" >
                {
                    sessions.slice().reverse().map((session, index) => (
                        <Accordion.Item key={session.session_path} eventKey={`${index}`}>
                            <Accordion.Header>{session.session_date}</Accordion.Header>
                            <Accordion.Body className="p-1" >
                                <RecordingTable recordings={session.recordings} participant={participant} session_date={session.session_date} imported={session.imported} callback={callback} />
                            </Accordion.Body>
                        </Accordion.Item>
                    ))
                }
            </Accordion >
        );
    }
};

// Add the sortByDate prop
const OuterAccordion = ({ participants, callback, sortByDate }) => {

    if (sortByDate) {
        const allSessions = participants.flatMap(participant => participant.sessions);
        const uniqueDates = Array.from(new Set(allSessions.map(session => session.session_date))).sort().reverse();


        return (
            <Accordion>
                {uniqueDates.map((date, index) => (
                    <Accordion.Item eventKey={`${index}`}>

                        <Accordion.Header>
                            {date}
                        </Accordion.Header>

                        <Accordion.Body className="p-0">
                            <InnerAccordion
                                sessions={date}
                                participants={participants.filter(participant => participant.sessions.some(session => session.session_date === date))}
                                callback={callback}
                                sortByDate={sortByDate}
                            />
                        </Accordion.Body>

                    </Accordion.Item>
                ))}
            </Accordion>
        );
    } else {

        return (
            <Accordion>
                {participants.map((participant, index) => (
                    <Accordion.Item eventKey={`${index}`}>

                        <Accordion.Header>
                            {participant.name}
                        </Accordion.Header>

                        <Accordion.Body className="p-0" >
                            <InnerAccordion sessions={participant.sessions} participants={participant.name} callback={callback} sortByDate={sortByDate} />
                        </Accordion.Body>

                    </Accordion.Item>
                ))}
            </Accordion>

        );
    }
};



const AnalysisHome = () => {

    const [recordingDb, setRecordingDb] = useState([]);
    const { fetchRecordingDb, toggleProcess, runCalibration, processSession } = useContext(AcquisitionState);

    // Add a state variable to track the toggle state
    const [sortByDate, setSortByDate] = useState(true);

    const radios = [
        { name: 'By Date', value: true },
        { name: 'By Participant', value: false },
    ];

    const onToggleShouldProcess = async (participant, session_date, recordingKey, isChecked) => {
        console.log("toggle", participant, session_date, recordingKey, isChecked);
        await toggleProcess(participant, recordingKey, isChecked)

        // // Find the corresponding entry in the recordingDb and toggle the should_process flag
        // const participantIndex = recordingDb.findIndex((participantEntry) => participantEntry.name === participant);
        // const sessionIndex = recordingDb[participantIndex].sessions.findIndex((sessionEntry) => sessionEntry.session_date === session_date);
        // const recordingIndex = recordingDb[participantIndex].sessions[sessionIndex].recordings.findIndex((recordingEntry) => recordingEntry.filename === recordingKey);

        // // make a local deep copy of recordingDb to edit
        // var _recordingDb = JSON.parse(JSON.stringify(recordingDb));

        // console.log(recordingDb[participantIndex].sessions[sessionIndex].recordings[recordingIndex].should_process)
        // _recordingDb[participantIndex].sessions[sessionIndex].recordings[recordingIndex].should_process = isChecked;
        // console.log(_recordingDb[participantIndex].sessions[sessionIndex].recordings[recordingIndex].should_process)
        // setRecordingDb(_recordingDb);
    };

    const onCalibrate = async (participant, session_date, recordingKey, isCharuco) => {
        console.log("calibrate", participant, session_date, recordingKey);
        await runCalibration(participant, recordingKey, isCharuco);
    }

    const onProcessSession = async (participant, session_date, videoProject) => {
        console.log("process session", participant, session_date, videoProject);
        await processSession(participant, session_date, videoProject);
    }

    // Create variable with both callbacks
    const callbacks = {
        onProcessSession: onProcessSession,
        onToggleShouldProcess: onToggleShouldProcess,
        onCalibrate: onCalibrate
    }

    useEffect(() => {

        const fetchData = async () => {
            const recordings = await fetchRecordingDb();
            console.log("recordings", recordings);
            setRecordingDb(recordings);
        }

        fetchData();

    }, [fetchRecordingDb]);

    const resyncData = async () => {
        const recordings = await fetchRecordingDb();
        console.log("recordings", recordings);
        setRecordingDb(recordings);
    }

    const toggle = (val) => {
        val = val == "true";
        console.log("toggle", val);
        setSortByDate(val);
    }

    return (
        <div>
            <h1>Recordings</h1>
            <Container className="g-4 p-2" gap={3}>

                <ButtonGroup className="mb-3 p-2">
                    {radios.map((radio, idx) => (
                        <ToggleButton
                            key={idx}
                            id={`radio-${idx}`}
                            type="radio"
                            variant={idx % 2 ? 'outline-success' : 'outline-danger'}
                            name="radio"
                            value={radio.value}
                            checked={sortByDate == radio.value}
                            onChange={(e) => toggle(e.currentTarget.value)}
                        >
                            {radio.name}
                        </ToggleButton>
                    ))}
                </ButtonGroup>
                <OuterAccordion participants={recordingDb} callback={callbacks} sortByDate={sortByDate} />
            </Container>
            <Button onClick={resyncData}>Refresh Video Database</Button>
        </div>
    );
};

export default AnalysisHome;