# This is the build file for the docker. Note this should be run from the
# parent directory for the necessary files to be available

.PHONY: clean build run

DIR := ${CURDIR}

build:
	docker build -t peabody124/mocap -f ./docker/Dockerfile .

run:
	docker run  -it --network=host -e REACT_APP_BASE_URL="localhost" -v /home/acadia/Documents/247cam/data/YFADatafile/VIDEO/:/data -v /home/acadia/Documents/247cam/camera_configs:/configs -v /etc/localtime:/etc/localtime:ro -v /home/acadia/Documents/247cam/datajoint_external:/datajoint_external peabody124/mocap 
