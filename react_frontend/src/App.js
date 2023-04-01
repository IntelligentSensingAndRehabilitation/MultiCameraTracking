// import logo from './logo.svg';
import './App.css';

// This is currently requiring 
// npm install react-router-dom@5.2.0 react-router-bootstrap@0.25.0
// if upgrading, then Switch changes to Routes (I think)
import { MemoryRouter as Router, Switch, Route } from 'react-router-dom';
import AcquisitionHome from './AcquisitionHome';
import Container from "react-bootstrap/Container";
import Button from "react-bootstrap/Button";
import ButtonToolbar from "react-bootstrap/ButtonToolbar";

import { Row } from "react-bootstrap";

//import { Navbar, Nav } from 'react-bootstrap';
import { LinkContainer } from 'react-router-bootstrap';

const Home = () => <span>Home</span>;

const About = () => <span>About</span>;

const Users = () => <span>Users</span>;

function App() {


  return (<Router>

    <Container className="p-3">
      <Row>
        <h2>
          Navigate to{" "}
          <ButtonToolbar className="custom-btn-toolbar">
            <LinkContainer to="/">
              <Button>Home</Button>
            </LinkContainer>
            <LinkContainer to="/about">
              <Button>About</Button>
            </LinkContainer>
            <LinkContainer to="/users">
              <Button>Users</Button>
            </LinkContainer>
          </ButtonToolbar>
        </h2>
      </Row>

      <Container className="p-5 mb-4 bg-light rounded-3">
        <h1 className="header">Welcome To React-Bootstrap</h1>
        <h2>
          Current Page is{" "}
          <Switch>
            <Route path="/about">
              <About />
            </Route>
            <Route path="/users">
              <Users />
            </Route>
            <Route path="/">
              <AcquisitionHome />
            </Route>
          </Switch>
        </h2>

      </Container>
    </Container>
  </Router>

  );
}

export default App;
