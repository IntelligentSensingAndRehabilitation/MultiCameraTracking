import PySpin
import time
import numpy as np
import simple_pyspin
from simple_pyspin import Camera
import pandas as pd

# Initialize system
system = PySpin.System.GetInstance()

# Get camera list
cam_list = simple_pyspin.list_cameras()

availability = {"in_use": [],
                "available": []}

for i, c in enumerate(cam_list):
    n = c.GetTLDeviceNodeMap()
    ni = PySpin.CStringPtr(n.GetNode("DeviceSerialNumber"))
    serial_number = ni.GetValue()

    print(f"({(i+1):02d}) Serial: {serial_number}")

    try:
        Camera(serial_number, lock=True).init()

        availability["available"].append(serial_number)
        print("    * Available")
    except Exception as e:
        availability["in_use"].append(serial_number)
        # print(e)
        print("    * In Use")

print(f"Currently {len(availability['in_use'])} cameras in use and {len(availability['available'])} available.")
print(availability)

# cams = [Camera(i, lock=True) for i in range(cam_list.GetSize())]
#
# for c in cam_list:
#     n = c.GetTLDeviceNodeMap()
#     ni = PySpin.CStringPtr(n.GetNode("DeviceSerialNumber"))
#     print(ni.GetValue())
#     cam_attributes[ni.GetValue()] = {}
#     with simple_pyspin.Camera(ni.GetValue()) as cam:
#         # print(cam.get_info('DeviceSerialNumber'))
#         # print(cam.get_info('TriggerMode'))
#         # print(cam.get_info('ExposureTime'))
#         # print(cam.get_info('AcquisitionFrameRate'))
#         # print(cam.get_info('TimestampLatchValue'))
#         # print(type(cam.get_info('TimestampLatchValue')))
#         # print(cam.get_info('TimestampLatchValue').keys())
#         # print(len(cam.camera_attributes))
#         # print(cam.camera_attributes.keys())
#
#         for attribute in cam.camera_attributes.keys():
#             try:
#                 # print(attribute)
#                 # print(cam.get_info(attribute))
#                 current_attribute = cam.get_info(attribute)
#
#                 if 'value' in current_attribute.keys():
#                     val = 'value'
#                 else:
#                     val = 'access'
#
#                 # print(type(current_attribute))
#                 # print(current_attribute['value'])
#                 cam_attributes[ni.GetValue()][attribute] = current_attribute[val]
#             except Exception as e:
#
#                 print(attribute)
#                 # print(current_attribute.keys())
#                 # print(current_attribute)
#                 # print("EXCEPTION: ",e)
#         del cam