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
import { AquisitionApi } from './AcquisitionApi';
import BiomechanicalReconstruction from './components/Visualizer';
import BiomechanicsBrowser from './components/BiomechanicsBrowser';
import Container from "react-bootstrap/Container";

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
            <LinkContainer to="/annotator">
              <Nav.Link>Annotate</Nav.Link>
            </LinkContainer>
            <LinkContainer to="/biomechanics_browser">
              <Nav.Link>Biomechanics</Nav.Link>
            </LinkContainer>
            <LinkContainer to="/visualize_biomechanics">
              <Nav.Link>Visualize Biomechanics</Nav.Link>
            </LinkContainer>
          </Nav>
        </Container>
      </Navbar>

      <AquisitionApi>
        <Routes>
          <Route path="/" element={<AcquisitionHome />} />
          <Route path="/analysis" element={<AnalysisHome />} />
          <Route path="/annotator" element={<BiomechanicalReconstruction data="true" />} />
          <Route path="/biomechanics_browser" element={<BiomechanicsBrowser />} />
          <Route path="/visualize_biomechanics" element={<BiomechanicalReconstruction data="false" />} />
        </Routes>
      </AquisitionApi>

    </div>

  );
}

export default App;
