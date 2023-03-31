// import logo from './logo.svg';
import './App.css';
import { Container } from 'react-bootstrap';
import Config from './components/Config';
import Participant from './components/Participant';
import RecordingInfo from './components/RecordingInfo';
import RecordingControl from './components/RecordingControl';
import Video from './components/Video';
import CameraStatusTable from './components/CameraStatusTable';
import PriorRecordingsTable from './components/PriorRecordingsTable';
import { AquisitionApi } from './AcquistionApi';

function App() {


  return (

    <div className="App">
      <h1 className="bg-secondary mb-4">Multi-camera Video Acquisition System</h1>

      <Container style={{ maxWidth: "1000px" }}>
        <AquisitionApi>
          <Config />
          <Participant />
          <RecordingInfo />
          <RecordingControl />
          <Video />
          <PriorRecordingsTable />
          <CameraStatusTable />
        </AquisitionApi>
      </Container>

    </div >
  );
}

export default App;
