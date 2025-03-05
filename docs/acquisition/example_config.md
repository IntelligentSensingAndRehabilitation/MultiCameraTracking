# Example config.yaml file

Config files should follow the format below. `camera_serial_number` should be updated with the serial numbers of the cameras that should be used for acquisition.


```
---
 camera-info:
   camera_serial_number (ex: 23336091):
     lens_info: "F1.4/6mm"
   camera_serial_number:
     lens_info: "F1.4/6mm"
   .
   .
   .
   camera_serial_number:
     lens_info: "F1.4/6mm"

 acquisition-type: 'max-frame' # 'max-frame' or 'continuous'

 acquisition-settings:
    exposure_time: 15000 # in microseconds
    frame_rate: 30
    video_segment_len: 1000 # if acquisition-type is 'continuous', this is the number of frames to record before starting a new file
    chunk_data: ['FrameID','SerialData']

# To do max-frame recording or continuous recording with a single start trigger, set line0 to 'Off'
# To do continuous recording, with each frame triggered, set line0 to 'ArduinoTrigger'
# To have 3.3V toggled when acquisition is started/stopped, set line2 to '3V3_Enable'
# To have the cameras receive serial data, set line3 to 'SerialOn'
 gpio-settings:
    line0: 'Off' # Opto-isolated input; the options for this pin are 'Off' 'ArduinoTrigger'
    line1: 'Off' # Opto-isolated output; the options for this pin are 'Off' or 'ExposureActive'
    line2: 'Off' # Non-isolated input/output; the options for this pin are 'Off' or '3V3_Enable'
    line3: 'Off' # Non-isolated input; the options for this pin are 'Off' or 'SerialOn'

# Any additional meta information can be added here
 meta-info:
   system: "Mobile"
   location: "Lab Space 3"
```
