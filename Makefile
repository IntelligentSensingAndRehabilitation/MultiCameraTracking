# This is the build file for the docker. Note this should be run from the
# parent directory for the necessary files to be available

.PHONY: clean build run

DIR := ${CURDIR}

build:
	docker build -t peabody124/mocap -f ./docker/Dockerfile .

run:
	docker run  -it --network=host -e REACT_APP_BASE_URL="localhost" -v /data:/data -v /camera_configs:/configs -v /etc/localtime:/etc/localtime:ro -v /mnt/datajoint_external:/datajoint_external peabody124/mocap /Mocap/start_acquisition_gui.sh

test:
	docker run --rm -it --network=host -v /data:/data -v /camera_configs:/configs -v /etc/localtime:/etc/localtime:ro -v /datajoint_external:/datajoint_external -v $(DIR)/tests:/Mocap/tests peabody124/mocap /Mocap/run_tests.sh

reset:
	docker run --rm -it --network=host -v /camera_configs:/configs -v /etc/localtime:/etc/localtime:ro peabody124/mocap /Mocap/run_reset.sh
