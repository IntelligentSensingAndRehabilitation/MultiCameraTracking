import { useState, useContext } from "react";
import { Row, Image } from "react-bootstrap";
import { AcquisitionState } from "../AcquistionApi";


const Video = () => {
    const { videoUrl } = useContext(AcquisitionState);

    // Set the image source to the video stream endpoint
    const videostream_src = `{API_BASE_URL}/video`;

    return (
        <Row md={10} className="g-4 p-2">
            <Image id="video_stream" src={videoUrl} rounded responsive />
        </Row>
    );

};

export default Video;