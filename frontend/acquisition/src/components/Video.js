import React from 'react';
import { useState, useContext, useRef, useCallback } from "react";
import { Row, Image } from "react-bootstrap";
import Accordion from 'react-bootstrap/Accordion';
import { AcquisitionState, useEffectOnce } from "../AcquisitionApi";


const Video = () => {
    const { videoUrl, isPreview, selectedCamera, selectCamera, cameraStatusList } = useContext(AcquisitionState);
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

    const handleImageClick = useCallback((e) => {
        if (!isPreview || selectedCamera !== null) return;
        const rect = e.target.getBoundingClientRect();
        const relX = (e.clientX - rect.left) / rect.width;
        const relY = (e.clientY - rect.top) / rect.height;
        const numCameras = cameraStatusList.length;
        if (numCameras === 0) return;
        const gridWidth = Math.ceil(Math.sqrt(numCameras));
        const gridHeight = Math.ceil(numCameras / gridWidth);
        const col = Math.floor(relX * gridWidth);
        const row = Math.floor(relY * gridHeight);
        const index = row * gridWidth + col;
        if (index < numCameras) {
            selectCamera(index);
        }
    }, [isPreview, selectedCamera, cameraStatusList, selectCamera]);

    const handleClose = useCallback(() => {
        selectCamera(null);
    }, [selectCamera]);

    const showClickable = isPreview && selectedCamera === null;

    return (
        <Accordion defaultActiveKey="0" className="g-4 p-2">
            <Accordion.Item eventKey="0">
                < Accordion.Header > Video Preview</Accordion.Header >
                <Accordion.Body>

                    <Row md={10} className="g-4 p-2">
                        <div style={{ position: 'relative', display: 'inline-block' }}>
                            <Image
                                id="video_stream"
                                src={imageSrc}
                                rounded
                                onClick={handleImageClick}
                                style={{ cursor: showClickable ? 'pointer' : 'default', width: '100%' }}
                            />
                            {isPreview && selectedCamera !== null && (
                                <button
                                    onClick={handleClose}
                                    style={{
                                        position: 'absolute',
                                        top: '10px',
                                        right: '25px',
                                        background: 'rgba(0, 0, 0, 0.6)',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '50%',
                                        width: '36px',
                                        height: '36px',
                                        fontSize: '18px',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        lineHeight: '1',
                                    }}
                                    aria-label="Close zoom"
                                >
                                    ✕
                                </button>
                            )}
                        </div>
                    </Row>
                </Accordion.Body>
            </Accordion.Item >
        </Accordion >
    );
};

export default Video;
