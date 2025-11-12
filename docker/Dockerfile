FROM ubuntu:22.04

### Install Spinnaker

COPY docker/flir/spinnaker-4.2.0.46-amd64 /spinnaker
COPY docker/flir/*.whl /spinnaker


ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y --no-install-recommends libusb-1.0-0 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
    curl screen python3 python3-pip python3-setuptools git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
    libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev vim libgtk2.0-dev libgtk-3-dev build-essential \
    cmake mysql-client ffmpeg

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    echo libgentl libspinnaker/accepted-flir-eula boolean true | debconf-set-selections && \
    dpkg -i /spinnaker/*.deb

RUN pip3 install /spinnaker/*.whl

### Install items needed for frontend (Node.js, npm)

RUN apt-get install -y ca-certificates curl gnupg
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
RUN apt-get update && apt-get install nodejs -y

RUN node -v
RUN npm -v

WORKDIR /Mocap/react_frontend

COPY react_frontend/package*.json /Mocap/react_frontend/

RUN npm install

COPY react_frontend /Mocap/react_frontend

### Installing extra things (curl, screen, netstat) (can be removed after image is in stable state)
RUN apt-get install -y net-tools

### Install EasyMocap
WORKDIR /
RUN git clone https://github.com/peabody124/EasyMocap
RUN pip install --upgrade pip
RUN pip install -e /EasyMocap

RUN git clone https://github.com/IntelligentSensingAndRehabilitation/PosePipeline.git
RUN pip install -e /PosePipeline

RUN apt-get install -y libtbb-dev
RUN pip3 install tbb
RUN pip3 install --upgrade tbb
RUN pip3 install numba

### Install MultiCameraTracking package and dependencies

WORKDIR /Mocap
# COPY requirements.txt .
# RUN pip3 install -r requirements.txt
COPY pyproject.toml README.md /Mocap/
COPY multi_camera /Mocap/multi_camera
COPY model_data /Mocap/model_data
RUN pip3 install -e /Mocap/

COPY docker/start_acquisition_gui.sh /Mocap/start_acquisition_gui.sh
RUN chmod +x /Mocap/start_acquisition_gui.sh

WORKDIR /
RUN mkdir -p /data
RUN mkdir -p /configs

# A hack currently, to have a local .datajoint_config.json replace the default one from posepipe
COPY datajoint_config.json /root/.datajoint_config.json

COPY tests /Mocap/tests
COPY run_tests.sh /Mocap/run_tests.sh

RUN pip3 install pytest

RUN pip install --upgrade numpy
RUN pip install --upgrade opencv-python opencv-contrib-python

COPY run_reset.sh /Mocap/run_reset.sh

# # Run bash script to start front end and back end
ENTRYPOINT ["/bin/bash"]
