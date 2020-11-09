"""
WebcamRecorder: This file includes every configuration with global usage. 

@authors:   Arno Schiller (AS)
@email:     schiller@swms.de
@version:   v0.0.3 
@license:   https://github.com/ArnoSchiller/DIH4CPS-PYTESTS

VERSION HISTORY
Version:    (Author) Description:                                   Date:
v0.0.1      (AS) First initialize. Added important configurations.  10.08.2020\n
v0.0.2      (AS) Added configurations for data_acquisition.         14.10.2020\n
v0.0.3      (AS) Restructured configs, movednot needed statements   04.11.2020\n
                to the bottom.                                                \n
"""

import os

global_user_name                = "test2"

# if the nunki board is used, the connection will be added in the init_videoCapture function (see webcam_capture) 

## ON BOARD CAMERA 
#global_camera_connection    = 0 

## CHINA CAMERA MODULE
#global_camera_connection    = "rtsp://admin:admin@192.168.3.70:8554"
#global_camera_connection    = "rtsp://admin:admin@192.168.8.22:8554"

## AXIS camera 
#global_camera_connection     = # "http://root:root@192.168.8.136/mjpg/1/video.mjpg"
global_camera_connection    = "http://root:root@192.168.178.78/mjpg/1/video.mjpg"

# Cloud 
global_cloud_access_key         = "minio"
global_cloud_secret_key         = "miniostorage"
global_cloud_url                = "https://minio.dih4cps.swms-cloud.com:9000/"

global_cloud_bucket_name        = "test-dih4cps"
global_trained_model_bucket     = "trained-models-dih4cps"
global_created_model_bucket     = "created-models-dih4cps"
# MQTT
global_mqtt_usinglocalhost      = False
global_mqtt_host                = "demo2.iotstack.co"
global_mqtt_user_name           = "pubclient"
global_mqtt_password            = "tiguitto"
global_mqtt_port                = 8883

### Maybe not usefull anymore
global_with_video_display       = False # True
global_with_video_process       = False # True
global_with_job_timer           = False # True
global_with_direct_rec          = True
global_use_light                = False

# Recorder 
global_camera_fps               = 20
global_recordsDir_name          = "Recordings"
global_recordsDir_path          = os.path.join("..", global_recordsDir_name)         

global_record_frequency_s       = 10 * 60           # 10 minutes
global_record_video_length_s    = 10    
global_max_video_len_seconds    = 20
global_max_video_len_frames     = global_max_video_len_seconds * global_camera_fps

global_direct_record_length_s   = 60 * 60       # 1 hour

