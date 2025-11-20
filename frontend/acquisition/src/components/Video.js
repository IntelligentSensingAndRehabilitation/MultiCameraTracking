import React from 'react';
import { useState, useContext, useRef } from "react";
import { Row, Image } from "react-bootstrap";
import Accordion from 'react-bootstrap/Accordion';
import { AcquisitionState, useEffectOnce } from "../AcquisitionApi";


const Video = () => {
    const { videoUrl } = useContext(AcquisitionState);
    const [imageSrc, setImageSrc] = useState("");
    const ws = useRef(null);

    useEffectOnce(() => {

        console.log('Connecting to video websocket...');

        ws.current = new WebSocket(videoUrl);

        ws.current.onopen = () => {
            console.log("Video WebSocket connected");
        };

        ws.current.onmessage = (event) => {
            console.log("new image")
            const data = event.data;

            if (imageSrc) {
                URL.revokeObjectURL(imageSrc);
            }

            const blob = new Blob([data], { type: "image/jpeg" });
            const url = URL.createObjectURL(blob);
            setImageSrc(url);
        };

        ws.current.onclose = () => {
            console.log("Video WebSocket disconnected");
        };

        ws.current.onerror = (event) => {
            console.log("Video WebSocket error observed:", event);
        }

        return () => {
            if (ws.current) {
                ws.current.close();
            }
            if (imageSrc) {
                URL.revokeObjectURL(imageSrc);
            }
        };
    }, []);

    return (
        <Accordion defaultActiveKey="0" className="g-4 p-2">
            <Accordion.Item eventKey="0">
                < Accordion.Header > Video Preview</Accordion.Header >
                <Accordion.Body>

                    <Row md={10} className="g-4 p-2">
                        <Image id="video_stream" src={imageSrc} rounded />
                    </Row>
                </Accordion.Body>
            </Accordion.Item >
        </Accordion >
    );
};

export default Video;