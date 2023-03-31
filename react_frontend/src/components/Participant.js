import { useState, useContext } from "react";
import { Row, Col, Form, Button } from "react-bootstrap";
import { AcquisitionState } from "../AcquistionApi";

const Participant = () => {
    const { setParticipant } = useContext(AcquisitionState);



    const [validated, setValidated] = useState(false);

    const handleSubmit = (event) => {
        console.log("handleSubmit", event.target.elements.participant.value);
        event.preventDefault();
        setParticipant(event.target.elements.participant.value);
        setValidated(true)
    };

    return (
        <Form noValidate validated={validated} onSubmit={handleSubmit} className="p-2">
            <Form.Group controlId="participant" as={Row} >
                <Form.Label column sm={3}>Participant:</Form.Label>
                <Col sm={6}>
                    <Form.Control
                        required
                        type="text"
                        placeholder="Participant identifier"
                        defaultValue=""
                        aria-describedby="participantHelpBlock"
                    />
                </Col>
                <Col sm={3} className="d-flex justify-content-end">
                    <Button variant="primary" type="submit">
                        New Session
                    </Button>
                </Col>
            </Form.Group>
            {/* <Form.Text id="participantHelpBlock" muted>
                Participant identifier should be unique for each participant, and like p###.
            </Form.Text> */}
        </Form>
    );

};

export default Participant;