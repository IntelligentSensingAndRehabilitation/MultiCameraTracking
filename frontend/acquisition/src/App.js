// import logo from './logo.svg';
import './App.css';
import React from 'react';

// This is currently requiring
// npm install react-router-dom@5.2.0 react-router-bootstrap@0.25.0
// if upgrading, then Switch changes to Routes (I think)
import { Routes, Route } from 'react-router-dom';
import { Navbar, Nav } from 'react-bootstrap';
import { LinkContainer } from 'react-router-bootstrap';
import AcquisitionHome from './AcquisitionHome';
import AnalysisHome from './AnalysisHome';
import { AcquisitionApi } from './AcquisitionApi';
import Container from "react-bootstrap/Container";
import SmplBrowser from './components/SmplBrowser';

function App() {

  return (
    <div className="App">

      <Navbar bg="dark" variant="dark">
        <Container>
          <Navbar.Brand href="#home">Markerless Mocap</Navbar.Brand>
          <Nav className="me-auto">
            <LinkContainer to="/">
              <Nav.Link>Acquisition</Nav.Link>
            </LinkContainer>
            <LinkContainer to="/analysis">
              <Nav.Link>Analyze</Nav.Link>
            </LinkContainer>
            <LinkContainer to="/smpl_browser">
              <Nav.Link>SMPL</Nav.Link>
            </LinkContainer>
          </Nav>
        </Container>
      </Navbar>

      {process.env.REACT_APP_TEST_MODE === 'true' && (
        <div style={{
          background: '#e74c3c',
          color: 'white',
          textAlign: 'center',
          padding: '8px',
          fontWeight: 'bold',
          letterSpacing: '1px',
        }}>
          TEST MODE — Data is not being saved to the production database
        </div>
      )}

      <AcquisitionApi>
        <Routes>
          <Route path="/" element={<AcquisitionHome />} />
          <Route path="/analysis" element={<AnalysisHome />} />
          <Route path="/smpl_browser" element={<SmplBrowser />} />
        </Routes>
      </AcquisitionApi>

    </div>

  );
}

export default App;
