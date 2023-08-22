# This is the build file for the docker. Note this should be run from the
# parent directory for the necessary files to be available

.PHONY: clean build run

DIR := ${CURDIR}

build:
	docker build -t peabody124/mocap -f ./docker/Dockerfile .

run:
	docker run  -it --network=host -v /data:/data -v /etc/localtime:/etc/localtime:ro -v /datajoint_external:/datajoint_external peabody124/mocap 

