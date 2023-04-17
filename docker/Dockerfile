FROM peabody124/posepipe

RUN pip3 install git+https://github.com/zju3dv/EasyMocap
RUN pip3 install tabulate

# ####### Now set up PosePipe itself
WORKDIR /Mocap
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY setup.py README.md /Mocap/
COPY multi_camera /Mocap/multi_camera
RUN pip3 install -e /Mocap

COPY docs/gaitrite_comparison_writeup/populate_gaitrite_analysis.py /Mocap
COPY scripts/session_pipeline.py /Mocap
COPY scripts/mocap_openpose.py /Mocap

ENTRYPOINT ["/usr/bin/python3"]
CMD ["mocap_openpose.py"]

