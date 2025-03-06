# Example config.yaml file

Config files should follow the format below.

Things you should check/change before using the config file:
- Serial numbers (`21182016, 22343864, etc.` in the example below) should be updated with the serial numbers of the cameras that should be used for acquisition.
- `lens_info` is simply metainfo
- You can optionally add the `flip_image` field for a given camera. If you do not include the `flip_image` field, or if you have `flip_image: False`, the image will not be flipped
- Most use cases utilize `max-frame` for `acquisition-type`. `max-frame` will record until `N frames` (set in the GUI) are recorded or the stop button is pressed. `continuous` will only stop if the stop button is pressed. `continuous` will record for `N=video_segment_len` frames and then start a new video segment and continue recording.
- In `acquisition-settings`, generally only `video_segment_len` and `chunk_data` are modified. If you don't need `chunk data`, you can set it to `[]`
- Set relevant `gpio-settings`
- Add any additional `meta-info`


```
---
 camera-info:
   21182016:
     lens_info: "F1.4/6mm"
   22343864:
     lens_info: "F1.6/4.4-11mm"
   22343863:
     lens_info: "F1.4/6mm"
   21132272:
     lens_info: "F1.4/6mm"
     flip_image: True
   22047081:
     lens_info: "F1.4/6mm"
     flip_image: False

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
