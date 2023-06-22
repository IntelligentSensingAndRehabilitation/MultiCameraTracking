#!/bin/env sh

# Start backend
python3 -m multi_camera.backend.fastapi &

cd /Mocap/react_frontend && npm start 