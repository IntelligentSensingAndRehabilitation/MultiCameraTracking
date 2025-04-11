#!/bin/env sh

mkdir -p flir
wget -nc -O flir/spinnaker-4.2.0.46-amd64-22.04-pkg.tar.gz https://flir.netx.net/file/asset/68771/original/attachment
wget -nc -O flir/spinnaker_python-4.2.0.46-cp310-cp310-linux_x86_64-22.04.tar.gz https://flir.netx.net/file/asset/68770/original/attachment

# unzip the files
tar -xzf flir/spinnaker-4.2.0.46-amd64-22.04-pkg.tar.gz -C flir
tar -xzf flir/spinnaker_python-4.2.0.46-cp310-cp310-linux_x86_64-22.04.tar.gz -C flir