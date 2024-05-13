#!/bin/bash

# Set MTU to max
chmod +x /home/cbm/MultiCameraTracking/set_mtu.sh
/home/cbm/MultiCameraTracking/set_mtu.sh

# Open a screen session and run the docker container
gnome-terminal -- bash -c 'screen -S multi_camera -m bash -c "cd /home/cbm/MultiCameraTracking && make run; exec bash"'

# Wait for front and back ends to load
sleep 5

# Open the web page
firefox http://localhost:3000

