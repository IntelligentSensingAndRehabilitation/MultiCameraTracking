import PySpin
import time
import numpy as np
import simple_pyspin
import pandas as pd

# Initialize system
system = PySpin.System.GetInstance()

# Get camera list
cam_list = simple_pyspin.list_cameras()

cam_attributes = {}

for c in cam_list:
    n = c.GetTLDeviceNodeMap()
    ni = PySpin.CStringPtr(n.GetNode("DeviceSerialNumber"))
    print(ni.GetValue())
    cam_attributes[ni.GetValue()] = {}
    with simple_pyspin.Camera(ni.GetValue()) as cam:
        # print(cam.get_info('DeviceSerialNumber'))
        # print(cam.get_info('TriggerMode'))
        # print(cam.get_info('ExposureTime'))
        # print(cam.get_info('AcquisitionFrameRate'))
        # print(cam.get_info('TimestampLatchValue'))
        # print(type(cam.get_info('TimestampLatchValue')))
        # print(cam.get_info('TimestampLatchValue').keys())
        # print(len(cam.camera_attributes))
        # print(cam.camera_attributes.keys())

        for attribute in cam.camera_attributes.keys():
            try:
                # print(attribute)
                # print(cam.get_info(attribute))
                current_attribute = cam.get_info(attribute)

                if 'value' in current_attribute.keys():
                    val = 'value'
                else:
                    val = 'access'

                # print(type(current_attribute))
                # print(current_attribute['value'])
                cam_attributes[ni.GetValue()][attribute] = current_attribute[val]
            except Exception as e:

                print(attribute)
                # print(current_attribute.keys())
                # print(current_attribute)
                # print("EXCEPTION: ",e)
        del cam
cam_df = pd.DataFrame(cam_attributes)

cam_df['unique_vals'] = cam_df.nunique(axis=1)
diff_val_df = cam_df[cam_df.unique_vals > 1]

print(cam_df)

import time
current_time = time.strftime('%Y%m%d_%H%M%S')
cam_df.to_csv(f'all_cam_settings_{current_time}.csv')
diff_val_df.to_csv(f'diff_cam_settings_{current_time}.csv')

# # Set up acquisition parameters
# num_buffers = 10
# packet_size_list = []
# packet_delay_list = []
#
# # Start acquisition for each camera
# for cam in cam_list:
#     cam.Init()
#     cam.BeginAcquisition()
#
#     nodemap = cam.GetTLDeviceNodeMap()
#     node_serial = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
#     serial_number = node_serial.GetValue()
#     print('Camera serial number: {}'.format(serial_number))
#
#     num_buffers = 10
#     packet_size_list = []
#     packet_delay_list = []
#
#     # Acquire buffers and calculate packet size and delay
#     for i in range(num_buffers):
#         t_start = time.time()
#         buffer = cam.GetNextImage()
#         t_end = time.time()
#         packet_size = buffer.GetImageSize()
#         packet_delay = (t_end - t_start) * 1000
#         packet_size_list.append(packet_size)
#         packet_delay_list.append(packet_delay)
#         buffer.Release()
#
#     print("PACKET SIZE")
#     print(packet_size_list)
#     print("PACKET DELAY AVG")
#     print(np.mean(packet_delay_list))
#
#     cam.EndAcquisition()
#     cam.DeInit()
#
# # Compare packet size and delay across all cameras
# if len(set(packet_size_list)) == 1:
#     print("All cameras have the same packet size.")
# else:
#     print(packet_size_list)
#
# if len(set(packet_delay_list)) == 1:
#     print("All cameras have the same packet delay.")
# else:
#     print(packet_delay_list)
#
# # Release system and cameras
# del cam
# cam_list.Clear()
# system.ReleaseInstance()