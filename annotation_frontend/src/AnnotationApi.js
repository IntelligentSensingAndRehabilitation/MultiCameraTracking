import React from 'react';
import { useState, useEffect, useRef, createContext } from 'react';
import axios from 'axios';

// get first part of base url from environment variable
// if not set, then use localhost
const BASE_HOSTNAME = process.env.REACT_APP_BASE_URL || 'localhost';

const BASE_URL = `${BASE_HOSTNAME}:8005/api/v1`;
const API_BASE_URL = `http://${BASE_URL}`;
const WS_BASE_URL = `ws://${BASE_URL}/ws`;


export const AnnotationState = createContext();

export const useEffectOnce = (effect) => {

    const destroyFunc = useRef();
    const effectCalled = useRef(false);
    const renderAfterCalled = useRef(false);
    const [, setVal] = useState(0);

    if (effectCalled.current) {
        renderAfterCalled.current = true;
    }

    useEffect(() => {

        // only execute the effect first time around
        if (!effectCalled.current) {
            destroyFunc.current = effect();
            effectCalled.current = true;
        }

        // this forces one render after the effect is run
        setVal(val => val + 1);

        return () => {
            // if the comp didn't render since the useEffect was called,
            // we know it's the dummy React cycle
            if (!renderAfterCalled.current) { return; }
            if (destroyFunc.current) {
                console.log(destroyFunc.current);
                destroyFunc.current();
            }
        };
    }, []);
};

export const AnnotationApi = (props) => {

    useEffectOnce(() => {

        var client_id = Date.now()

        const url = `${WS_BASE_URL}/${client_id}`;
        const socket = new WebSocket(url);

        console.log("Connecting to websocket..." + url)

        socket.onopen = (event) => {
            console.log("WebSocket connection established" + url, event);
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("WebSocket message received", data);
        };

        socket.onclose = (event) => {
            console.log("WebSocket connection closed" + url, event);
        };

        socket.onerror = (event) => {
            console.log("WebSocket error observed" + url + ":", event);
        }

        //clean up function when we close page
        return () => {
            console.log("Closing WebSocket " + url + " ...")
            socket.close();
        }
    }, []);

    // Mesh functions

    const fetchUnannotatedRecordings = async () => {
        const recordings = await axios.get(`${API_BASE_URL}/unannotated_recordings`);
        return recordings.data.video_base_filenames;
    };

    const annotateRecording = async (filename, ids) => {
        const response = await axios.post(`${API_BASE_URL}/annotation`,
            {
                video_base_filename: filename,
                ids: ids
            }
        );
        return response.data.success;
    };

    async function fetchMesh(filename, downsampling) {
        // Fetch the mesh data for the given recording
        const response = await axios.get(`${API_BASE_URL}/mesh`, {
            params: {
                filename: filename,
                downsample: downsampling
            }
        });
        const data = response.data;
        // unpack the base64 encoded mesh data
        data.meshes = JSON.parse(Buffer.from(data.meshes, 'base64'))
        return response.data;
    }

    return (<AnnotationState.Provider value={{
        videoUrl: `ws://${BASE_URL}/video_ws`,
        meshUrl: `ws://${BASE_URL}/mesh_ws`,
        fetchUnannotatedRecordings,
        annotateRecording,
        fetchMesh,
    }}> {props.children} </AnnotationState.Provider >)
};
