import React from 'react';
import { useState, useEffect, createContext } from 'react';
import axios from 'axios';

const BASE_URL = 'localhost:8000/api/v1';
const API_BASE_URL = `http://${BASE_URL}`;
const WS_BASE_URL = `ws://${BASE_URL}/ws`;

const initialState = {
    partipant: "",
    recordingDir: "",
    recordingFilename: "",
    availableConfigs: [],
}

export const AcquisitionState = createContext(initialState);

export const AquisitionApi = (props) => {

    const [participant, setParticipant] = useState("");
    const [cameraStatusList, setCameraStatusList] = useState([]);
    const [availableConfigs, setAvailableConfigs] = useState([]);
    const [currentConfig, setCurrentConfig] = useState('');
    const [priorRecordings, setPriorRecordings] = useState([]);
    const [recordingSystemStatus, setRecordingSystemStatus] = useState(null);
    const [recordingDir, setRecordingDir] = useState('');
    const [recordingFileBase, setRecordingFileBase] = useState('');
    const [recordingFilename, setRecordingFilename] = useState('');

    // useEffect(() => {
    //     axios.interceptors.request.use(request => {
    //         console.log('Starting Request', JSON.stringify(request, null, 2))
    //         return request
    //     })

    //     axios.interceptors.response.use(response => {
    //         console.log('Response:', JSON.stringify(response, null, 2))
    //         return response
    //     })
    // }, []);

    useEffect(() => {

        const socket = new WebSocket(WS_BASE_URL);

        console.log("Connecting to websocket...")

        socket.onopen = (event) => {
            console.log("WebSocket connection established", event);
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("WebSocket message received", data);
            setRecordingSystemStatus(data.status);
        };

        socket.onclose = (event) => {
            console.log("WebSocket connection closed", event);
        };

        socket.onerror = (event) => {
            console.log("WebSocket error observed:", event);
        }

        //clean up function when we close page
        return () => socket.close();
    }, []);

    useEffect(() => {
        fetchCameraStatus();
        fetchConfigs();
        fetchRecordings();
        fetchCurrentConfig();
        fetchRecordingStatus();
        fetchSession();
    }, []);

    useEffect(() => {
        updateConfig();
    }, [currentConfig]);

    useEffect(() => {
        console.log("recordingDir: " + recordingDir);
        console.log("recordingFileBase: " + recordingFileBase);
        console.log("recordingFilename: " + recordingFilename);
    }, [recordingDir, recordingFileBase, recordingFilename]);

    async function fetchSession() {
        const response = await axios.get(`${API_BASE_URL}/session`);
        const data = response.data;

        setParticipant(data.participant_name);
        setRecordingDir(data.recording_path);
        setRecordingFileBase(data.participant_name);
    }

    async function newSession(participant) {
        if (participant && participant.length > 0) {
            console.log("Creating new session for participant: ", participant);
            const response = await axios.post(`${API_BASE_URL}/session`, null,
                {
                    headers: {
                        'Content-Type': 'application/json',
                    }, params:
                    {
                        subject_id: participant,
                    }
                });

            const data = response.data;
            console.log(data);

            setParticipant(data.participant_name);
            setRecordingDir(data.recording_path);
            setRecordingFileBase(data.participant_name);
        }
    }

    async function newTrial(comment, max_frames) {

        // set max frames to 100 if undefined
        if (max_frames === undefined) {
            max_frames = 100;
        }

        if (participant && participant.length > 0) {
            console.log("Starting recording for participant: ", participant);
            const response = await axios.post(`${API_BASE_URL}/new_trial`,
                {
                    recording_dir: recordingDir,
                    recording_filename: recordingFileBase,
                    comment: comment,
                    max_frames: max_frames
                });

            const data = response.data;
            console.log(data);
            setRecordingFilename(data.recording_file_name);
        }
    }

    async function calibrationVideo(max_frames) {
        if (participant && participant.length > 0) {
            console.log("Creating new session for participant: ", participant);
            const response = await axios.post(`${API_BASE_URL}/new_trial`,
                {
                    recording_dir: recordingDir,
                    recording_filename: "calibration",
                    comment: "calibration",
                    max_frames: max_frames
                });

            const data = response.data;
            console.log(data);
            setRecordingFilename(data.recording_file_name);
        }
    }

    async function previewVideo(max_frames) {
        await axios.post(`${API_BASE_URL}/preview`);
    }

    async function stopAcquisition() {
        await axios.post(`${API_BASE_URL}/stop`);
    }

    const fetchCameraStatus = async () => {
        const response = await axios.get(`${API_BASE_URL}/camera_status`);
        setCameraStatusList(response.data);
    };

    const fetchRecordingStatus = async () => {
        const response = await axios.get(`${API_BASE_URL}/recording_status`);
        console.log("fetchRecordingStatus: ", response.data);
        setRecordingSystemStatus(response.data);
    };

    const fetchRecordings = async () => {
        const response = await axios.get(`${API_BASE_URL}/prior_recordings`);
        setPriorRecordings(response.data);
    };

    async function fetchRecordingDb() {
        const response = await axios.get(`${API_BASE_URL}/recording_db`);
        console.log("Recording DB: ", response.data)
        return response.data;
    };

    // Camera configuration settings

    const fetchConfigs = async () => {
        const response = await axios.get(`${API_BASE_URL}/configs`);
        setAvailableConfigs(response.data);
    };

    const fetchCurrentConfig = async () => {
        const response = await axios.get(`${API_BASE_URL}/current_config`);
        console.log("fetchCurrentConfig: ", response.data);
        setCurrentConfig(response.data);
    };

    const updateConfig = async () => {
        console.log("updateConfig: ", currentConfig);
        if (currentConfig) {
            console.log("Updating config: ", currentConfig);
            await axios.post(`${API_BASE_URL}/current_config`, { config: currentConfig });
            fetchCameraStatus();
        }
    };

    const resetCameras = async () => {
        console.log("resetCameras");
        await axios.post(`${API_BASE_URL}/reset_cameras`);
        fetchCameraStatus();
    };

    useEffect(() => {
        //Implementing the setInterval method
        //const interval = setInterval(() => {
        fetchCameraStatus()
        fetchRecordings()
        //}, 500);

        //Clearing the interval
        //return () => clearInterval(interval);
    }, [recordingSystemStatus, participant]);

    return (<AcquisitionState.Provider value={{
        participant: participant,
        recordingDir: recordingDir,
        recordingFileBase: recordingFileBase,
        recordingFilename: recordingFilename,
        availableConfigs: availableConfigs,
        currentConfig: currentConfig,
        cameraStatusList: cameraStatusList,
        priorRecordings: priorRecordings,
        videoUrl: `ws://${BASE_URL}/video_ws`,
        recordingSystemStatus: recordingSystemStatus,
        setCurrentConfig,
        resetCameras,
        newSession,
        newTrial,
        previewVideo,
        calibrationVideo,
        stopAcquisition,
        fetchRecordingDb
    }}> {props.children} </AcquisitionState.Provider >)
    //return (<div> {children} </div>)
};
