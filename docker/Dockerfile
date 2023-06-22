FROM peabody124/posepipe

# A hack currently, to have a local .datajoint_config.json replace the default one from posepipe
COPY datajoint_config.json /root/.datajoint_config.json


### Install Spinnaker

COPY docker/flir/spinnaker-3.1.0.79-amd64 /spinnaker
COPY docker/flir/*.whl /spinnaker


ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y libusb-1.0-0 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools curl screen

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    echo libgentl libspinnaker/accepted-flir-eula boolean true | debconf-set-selections && \
    dpkg -i /spinnaker/*.deb

# RUN /spinnaker/configure_gentl_paths.sh 64

RUN pip3 install /spinnaker/*.whl

# Install items needed for frontend

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs

RUN node -v
RUN npm -v

WORKDIR /Mocap/react_frontend

COPY react_frontend/package*.json /Mocap/react_frontend/

RUN npm install

COPY react_frontend /Mocap/react_frontend


EXPOSE 3000
EXPOSE 8000

# Installing extra things (curl, screen, netstat)
RUN apt-get install -y net-tools

# Now install other requirements for MMC system

# RUN pip3 install git+https://github.com/zju3dv/EasyMocap
# RUN pip3 install tabulate

# Now install MMC itself

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

COPY cotton_lab_config_20230620.yaml /Mocap


ENTRYPOINT ["/bin/bash"]
CMD ["/Mocap/start_acquisition_gui.sh"]


# CMD ["npm","start"]

# COPY docs/gaitrite_comparison_writeup/populate_gaitrite_analysis.py /Mocap
# COPY scripts/session_pipeline.py /Mocap
# COPY scripts/mocap_openpose.py /Mocap


# ENTRYPOINT ["/usr/bin/python3"]
#CMD ["-m","multi_camera.acquisition.flir_recording_api","test"]
# CMD ["-m","multi_camera.acquisition.cameras_reset","-a"]
# ENTRYPOINT ["/bin/bash"]