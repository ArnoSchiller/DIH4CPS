"""
WebcamRecorder: This file includes every configuration with global usage. 

@authors:   Arno Schiller (AS)
@email:     schiller@swms.de
@version:   v0.0.3 
@license:   https://github.com/ArnoSchiller/DIH4CPS-PYTESTS

VERSION HISTORY
Version:    (Author) Description:                                   Date:
v4.0.0      (AS) Initialize. See previous versions for more infos.  05-02-2021\n
v4.0.1      (AS) Removed parameters not needed and added context    05-02-2021\n
                informations, detection parameters and USB camera.            \n
"""

import os
import cv2

## contect informations
global_measurement          = "shrimps2"
global_user_name            = "jetsonNanoTests"
global_location             = "test_location"

## detection parameters
global_model_name           = "yolov5l_dsv1_aug"
score_thresh                = 0.5
iou_thresh                  = 0.5


# if the nunki board is used, the connection will be added in the init_videoCapture function (see webcam_capture) 

## ON BOARD CAMERA / USB camera
global_camera_connection    = 1 + cv2.CAP_DSHOW # USB camera under windows

## MQTT
global_mqtt_usinglocalhost      = False
global_mqtt_host                = "demo2.iotstack.co"
global_mqtt_user_name           = "pubclient"
global_mqtt_password            = "tiguitto"
global_mqtt_port                = 8883
