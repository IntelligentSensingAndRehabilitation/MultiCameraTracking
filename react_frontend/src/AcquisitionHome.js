import React from 'react';
import { Container } from 'react-bootstrap';
import Config from './components/Config';
import Participant from './components/Participant';
import RecordingInfo from './components/RecordingInfo';
import RecordingControl from './components/RecordingControl';
import Video from './components/Video';
import CameraStatusTable from './components/CameraStatusTable';
import PriorRecordingsTable from './components/PriorRecordingsTable';


const AcquisitionHome = () => {
    return (
        <Container>
            <Config />
            <Participant />
            <RecordingInfo />
            <RecordingControl />
            <Video />
            <PriorRecordingsTable />
            <CameraStatusTable />
        </Container>
    )
}

export default AcquisitionHome;