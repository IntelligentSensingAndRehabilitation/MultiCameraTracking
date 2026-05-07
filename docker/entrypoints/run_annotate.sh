#!/bin/env sh

# Start backend
python3 -m multi_camera.backend.fastapi_annotation &

cd /Mocap/frontend/annotation && PORT=3005 npm start