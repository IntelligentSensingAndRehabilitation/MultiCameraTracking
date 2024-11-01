#!/bin/env sh

# Start backend
python3 -m multi_camera.backend.fastapi_annotation &

cd /Mocap/annotation_frontend && npm start 