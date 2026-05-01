import React from 'react';
import { useState, useContext, useRef, useCallback } from "react";
import { Row, Image } from "react-bootstrap";
import Accordion from 'react-bootstrap/Accordion';
import { AcquisitionState, useEffectOnce } from "../AcquisitionApi";


const Video = () => {
    const { videoUrl, isPreview, selectedCamera, selectCamera, cameraStatusList } = useContext(AcquisitionState);
    const [imageSrc, setImageSrc] = useState("");
    const ws = useRef(null);
    const lastUrl = useRef("");
    const pendingUrl = useRef("");
    const rafId = useRef(0);

    useEffectOnce(() => {

        console.log('Connecting to video websocket...');

        ws.current = new WebSocket(videoUrl);

        ws.current.onopen = () => {
            console.log("Video WebSocket connected");
        };

        // Coalesce inbound frames into one React render per browser refresh.
        // Cameras stream at 30+ Hz; rendering each frame walks the full fiber
        // tree (especially expensive in dev mode with Strict Mode + DevTools)
        // and was causing the page to drown in allocations during preview.
        ws.current.onmessage = (event) => {
            const blob = new Blob([event.data], { type: "image/jpeg" });
            const url = URL.createObjectURL(blob);
            if (pendingUrl.current) {
                URL.revokeObjectURL(pendingUrl.current);
            }
            pendingUrl.current = url;
            if (rafId.current === 0) {
                rafId.current = requestAnimationFrame(() => {
                    rafId.current = 0;
                    const next = pendingUrl.current;
                    pendingUrl.current = "";
                    if (lastUrl.current) {
                        URL.revokeObjectURL(lastUrl.current);
                    }
                    lastUrl.current = next;
                    setImageSrc(next);
                });
            }
        };

        ws.current.onclose = () => {
            console.log("Video WebSocket disconnected");
        };

        ws.current.onerror = (event) => {
            console.log("Video WebSocket error observed:", event);
        }

        return () => {
            if (rafId.current !== 0) {
                cancelAnimationFrame(rafId.current);
                rafId.current = 0;
            }
            if (pendingUrl.current) {
                URL.revokeObjectURL(pendingUrl.current);
                pendingUrl.current = "";
            }
            if (ws.current) {
                ws.current.close();
            }
            if (lastUrl.current) {
                URL.revokeObjectURL(lastUrl.current);
                lastUrl.current = "";
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
