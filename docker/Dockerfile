FROM peabody124/posepipe

### Install Spinnaker

COPY docker/flir/spinnaker-3.1.0.79-amd64 /spinnaker
COPY docker/flir/*.whl /spinnaker


ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y libusb-1.0-0 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools curl screen

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    echo libgentl libspinnaker/accepted-flir-eula boolean true | debconf-set-selections && \
    dpkg -i /spinnaker/*.deb

RUN pip3 install /spinnaker/*.whl

### Install items needed for frontend (Node.js, npm)

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs

RUN node -v
RUN npm -v

WORKDIR /Mocap/react_frontend

COPY react_frontend/package*.json /Mocap/react_frontend/

RUN npm install

COPY react_frontend /Mocap/react_frontend

### Installing extra things (curl, screen, netstat) (can be removed after image is in stable state)
RUN apt-get install -y net-tools

### Install EasyMocap
RUN pip3 install git+https://github.com/peabody124/EasyMocap

RUN apt-get install -y libtbb-dev
RUN pip3 install tbb
RUN pip3 install --upgrade tbb
RUN pip3 install numba

### Install MultiCameraTracking package and dependencies

WORKDIR /Mocap
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY setup.py README.md /Mocap/
COPY multi_camera /Mocap/multi_camera
RUN pip3 install -e /Mocap/

WORKDIR /Mocap

COPY docker/start_acquisition_gui.sh /Mocap/start_acquisition_gui.sh
RUN chmod +x /Mocap/start_acquisition_gui.sh

WORKDIR /
RUN mkdir -p /data
COPY configs /configs

# A hack currently, to have a local .datajoint_config.json replace the default one from posepipe
COPY datajoint_config.json /root/.datajoint_config.json

# Run bash script to start front end and back end
ENTRYPOINT ["/bin/bash"]
CMD ["/Mocap/start_acquisition_gui.sh"]