import React, { useState, useEffect, useRef, useContext } from 'react';
import { Container, Row, Col, Dropdown } from "react-bootstrap";
import { Viewer } from './visualization_js/viewer.js';
import { AcquisitionState } from "../AcquisitionApi.js";

const system = {
    'meshes': {},
    'geoms': {},
    'smpl': {},
    'keypoints': null,
    'states': {},
    'dt': 0.033
}

const SmplBrowser = () => {

    const viewerRef = useRef();
    const containerRef = useRef();
    const guiRef = useRef();

    const { fetchSmplTrials, fetchSmpl } = useContext(AcquisitionState);

    const [biomechanicalRecordings, setBiomechanicalRecordings] = useState([]);
    const [participantIds, setParticipantIds] = useState([]);
    const [selectedParticipant, setSelectedParticipant] = useState(null);
    const [selectedSession, setSelectedSession] = useState(null);
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [recordingValidated, setRecordingValidated] = useState(false);

    useEffect(() => {
        fetchSmplTrials().then((data) => {

            console.log("SMPL recordings: ", data);
            setBiomechanicalRecordings(data);

            const participantIds = Array.from(new Set(data.map((recording) => recording.participant_id))).sort();

            setParticipantIds(participantIds);
            console.log("Participant IDs: ", participantIds)
        });

        return () => {
        }
    }, []);


    // Filter session dates based on selected participant
    const sessionDates =
        selectedParticipant !== null
            ? Array.from(
                new Set(
                    biomechanicalRecordings
                        .filter((recording) => recording.participant_id === selectedParticipant)
                        .map((recording) => recording.session_date)
                )
            )
            : [];

    // Filter video filenames based on selected participant and session date
    const videoFilenames =
        selectedParticipant !== null && selectedSession !== null
            ? biomechanicalRecordings
                .filter(
                    (recording) =>
                        recording.participant_id === selectedParticipant &&
                        recording.session_date === selectedSession
                )
                .map((recording) => recording.video_base_filename)
            : [];

    useEffect(() => {
        console.log("Available session dates: ", sessionDates);
        if (sessionDates.length) {
            setSelectedSession(sessionDates[0]);
        }
    }, [sessionDates]);

    useEffect(() => {
        console.log("Selected participant: ", selectedParticipant);
        console.log("Selected session: ", selectedSession);
        console.log("Selected video: ", selectedVideo);

        setRecordingValidated(false);
        closeViewer();

        if (selectedVideo != null && selectedVideo.length > 0) {
            fetchSmpl(selectedVideo).then((data) => {
                console.log("Biomechanics: ", data);
                setRecordingValidated(true);
                openViewer(data);
            });
        }
    }, [selectedParticipant, selectedSession, selectedVideo]);

    const closeViewer = () => {
        if (viewerRef.current != null) {
            viewerRef.current.close();
            // delete the viewerRef.current
            viewerRef.current = null;
        }
    }

    const openViewer = (smpl) => {

        console.log("SMPL data: ", smpl)
        system.smpl = smpl;

        const domElement = containerRef.current;
        const guiElement = guiRef.current;
        viewerRef.current = new Viewer(domElement, system, guiElement);

        setRecordingValidated(true);
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

    const dropDownStyle = {
        maxHeight: '28px',
        overflowY: 'scroll'
    }


    return (
        <Container>
            <Row>
                <Col>
                    <h3>Select a participant</h3>
                    <Dropdown
                        onSelect={(selectedKey) => {
                            setSelectedParticipant(selectedKey);
                            console.log("available sessions: ", sessionDates);
                            setSelectedSession(null);
                            setSelectedVideo(null);
                        }}>
                        <Dropdown.Toggle variant="outline-secondary" id="participant-dropdown"
                            style={{ paddingBottom: 5 }}>
                            {selectedParticipant || 'Select Participant'}
                        </Dropdown.Toggle>
                        <Dropdown.Menu style={{ overflowY: 'scroll', maxHeight: "250px", zIndex: 1000 }}>
                            {participantIds.map((id) => (
                                <Dropdown.Item key={id} eventKey={id}>
                                    {id}
                                </Dropdown.Item>
                            ))}
                        </Dropdown.Menu>
                    </Dropdown>
                </Col>
                <Col>
                    <h3>Select a session date</h3>
                    <Dropdown onSelect={(selectedKey) => {
                        setSelectedSession(selectedKey);
                        setSelectedVideo(null);
                    }} disabled={!selectedParticipant}>
                        <Dropdown.Toggle variant="outline-secondary" id="session-dropdown">
                            {selectedSession || 'Select Session'}
                        </Dropdown.Toggle>
                        <Dropdown.Menu>
                            {sessionDates.map((date) => (
                                <Dropdown.Item key={date} eventKey={date}>
                                    {date}
                                </Dropdown.Item>
                            ))}
                        </Dropdown.Menu>
                    </Dropdown>
                </Col>
                <Col>
                    <h3>Select a video</h3>
                    <Dropdown
                        onSelect={(selectedKey) => {
                            setSelectedVideo(selectedKey);
                            console.log('Selected video:', selectedKey);
                        }}
                        disabled={!selectedParticipant || !selectedSession}
                    >
                        <Dropdown.Toggle variant="outline-secondary" id="video-dropdown">
                            {selectedVideo || 'Select Video'}
                        </Dropdown.Toggle>
                        <Dropdown.Menu style={{ maxHeight: '200px', overflowY: 'scroll' }}>
                            {videoFilenames.map((filename) => (
                                <Dropdown.Item key={filename} eventKey={filename}>
                                    {filename}
                                </Dropdown.Item>
                            ))}
                        </Dropdown.Menu>
                    </Dropdown>
                </Col>
            </Row>

            <Row className='p-2'>
                <div ref={guiRef} id="gui" style={guiStyle}></div>
                <div ref={containerRef} id="brax-viewer" style={viewerStyle}></div>
            </Row>

        </Container >
    );
};

export default SmplBrowser;