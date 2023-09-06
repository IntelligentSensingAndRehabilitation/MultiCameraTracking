#!/bin/env sh

mkdir -p flir
wget -nc -O flir/spinnaker-3.1.0.79-amd64-pkg.tar.gz https://flir.netx.net/file/asset/54404/original/attachment
wget -nc -O flir/spinnaker_python-3.1.0.79-cp38-cp38-linux_x86_64.tar.gz https://flir.netx.net/file/asset/54407/original/attachment

# unzip the files
tar -xzf flir/spinnaker-3.1.0.79-amd64-pkg.tar.gz -C flir
tar -xzf flir/spinnaker_python-3.1.0.79-cp38-cp38-linux_x86_64.tar.gz -C flir