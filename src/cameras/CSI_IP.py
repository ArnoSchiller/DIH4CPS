import cv2
import threading

import sys
sys.path.append("./cameras")
from camera_interface import CameraTypes

def csi_gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class CSI_IP:
    'interface for CSI and IP cameras.'
    ImageCallback = None
    ImageCallbackData = None

    def __init__(self, camera_type:CameraTypes):
        ''' Constructor
        :return: none
        '''
        self.camera_type = camera_type

    def openDevice(self,connection, width, height, framerate):
        if not type(framerate) == float:
            framerate = float(framerate.split("/")[0])

        if self.camera_type == CameraTypes.CSI:
            self.capture = cv2.VideoCapture(csi_gstreamer_pipeline(framerate=framerate, display_width=width, display_height=height), cv2.CAP_GSTREAMER)
        else:
            self.capture = cv2.VideoCapture(connection)

    def grab_frames(self):
        while self.capture.isOpend():
            ret_val, img = self.capture.read()
            if ret_val:
                self.image = img
                self.ImageCallback(self, *self.ImageCallbackData)

    def Get_image(self):
        return self.image

    def Set_Image_Callback(self, function, *data):
        self.ImageCallback = function
        self.ImageCallbackData = data

    def Start_pipeline(self):
        if self.ImageCallback is None:
            print("CSI_IP: ImageCallback is not defined.")
            return False
        self.capture_thread = threading.Thread(self.grab_frames)
        self.capture_thread.start()

