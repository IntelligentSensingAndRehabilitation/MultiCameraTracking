import { useState, useEffect, createContext } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';
const WS_BASE_URL = 'ws://localhost:8000/api/v1/ws';

const initialState = {
    partipant: "",
    recordingDir: "",
    recordingFilename: "",
    availableConfigs: [],
}

export const AcquisitionState = createContext(initialState);

export const AquisitionApi = (props) => {

    const [participant, setParticipant] = useState([]);
    const [cameraStatusList, setCameraStatusList] = useState([]);
    const [availableConfigs, setAvailableConfigs] = useState([]);
    const [currentConfig, setCurrentConfig] = useState('');
    const [selectedConfig, setSelectedConfig] = useState('');
    const [priorRecordings, setPriorRecordings] = useState([]);
    const [recordingSystemStatus, setRecordingSystemStatus] = useState(null);
    const [recordingDir, setRecordingDir] = useState('');
    const [recordingFileBase, setRecordingFileBase] = useState('');
    const [recordingFilename, setRecordingFilename] = useState('');

    axios.interceptors.request.use(request => {
        console.log('Starting Request', JSON.stringify(request, null, 2))
        return request
    })

    axios.interceptors.response.use(response => {
        console.log('Response:', JSON.stringify(response, null, 2))
        return response
    })

    useEffect(() => {
        const socket = new WebSocket(WS_BASE_URL);

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
    }, []);

    useEffect(() => {
        fetchCameraStatus();
        fetchConfigs();
        fetchRecordings();
        fetchCurrentConfig();
        fetchRecordingStatus();
    }, []);

    useEffect(() => {
        updateConfig();
    }, [currentConfig]);

    useEffect(() => {
        newSession();
    }, [participant]);

    useEffect(() => {
        console.log("recordingDir: " + recordingDir);
        console.log("recordingFileBase: " + recordingFileBase);
        console.log("recordingFilename: " + recordingFilename);
    }, [recordingDir, recordingFileBase, recordingFilename]);

    async function newSession() {
        if (participant && participant.length > 0) {
            console.log("Creating new session for participant: ", participant);
            const response = await axios.post(`${API_BASE_URL}/new_session`, null,
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

            setRecordingDir(data.recording_dir);
            setRecordingFileBase(data.recording_filename);
        }
    }

    async function newTrial(comment) {
        if (participant && participant.length > 0) {
            console.log("Creating new session for participant: ", participant);
            const response = await axios.post(`${API_BASE_URL}/new_trial`,
                {
                    recording_dir: recordingDir,
                    recording_filename: recordingFileBase,
                    comment: comment
                });

            const data = response.data;
            console.log(data);
            setRecordingFilename(data.recording_file_name);
        }
    }

    async function calibrationVideo(comment) {
        if (participant && participant.length > 0) {
            console.log("Creating new session for participant: ", participant);
            const response = await axios.post(`${API_BASE_URL}/new_trial`,
                {
                    recording_dir: recordingDir,
                    recording_filename: "calibration",
                    comment: "calibration"
                });

            const data = response.data;
            console.log(data);
            setRecordingFilename(data.recording_file_name);
        }
    }

    async function previewVideo() {
        const response = await axios.post(`${API_BASE_URL}/preview`);
    }

    async function stopAcquisition() {
        const response = await axios.post(`${API_BASE_URL}/preview`);
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
        console.log("Prior recordings: ", response.data)
        setPriorRecordings(response.data);
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
            await axios.post(`${API_BASE_URL}/update_config`, { config: currentConfig });
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
        const interval = setInterval(() => {
            fetchCameraStatus()
            fetchRecordings()
        }, 5000);

        //Clearing the interval
        return () => clearInterval(interval);
    }, []);

    return (<AcquisitionState.Provider value={{
        partipant: participant,
        recordingDir: recordingDir,
        recordingFileBase: recordingFileBase,
        recordingFilename: recordingFilename,
        availableConfigs: availableConfigs,
        currentConfig: currentConfig,
        cameraStatusList: cameraStatusList,
        priorRecordings: priorRecordings,
        videoUrl: `${API_BASE_URL}/video`,
        recordingSystemStatus: recordingSystemStatus,
        setCurrentConfig,
        resetCameras,
        setParticipant,
        newTrial,
        previewVideo,
        calibrationVideo,
        stopAcquisition
    }}> {props.children} </AcquisitionState.Provider >)
    //return (<div> {children} </div>)
};
