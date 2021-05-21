import datetime
import time
import cv2 

from enum import Enum

import TIS, CSI_IP

class CameraTypes(Enum):
    TIS = 0
    CSI = 1
    IP = 2

    def fromString(s:str):
        if( s.lower() == "tis" ):
            return CameraTypes.TIS

        if( s.lower() == "csi" ):
            return CameraTypes.CSI
            
        if( s.lower() == "ip" ):
            return CameraTypes.IP

        return False

    def toString(pf):
        if( pf == CameraTypes.TIS ):
            return "TheImagingSource camera"

        if( pf == CameraTypes.CSI ):
            return "CSI camera"

        if( pf == CameraTypes.IP ):
            return "IP camera"

        return "NONE"

class SinkFormats(Enum):
    GRAY8 = 0
    GRAY16_LE = 1
    BGRA = 2

    def toString(pf):
        if( pf == SinkFormats.GRAY16_LE ):
            return "GRAY16_LE"

        if( pf == SinkFormats.GRAY8 ):
            return "GRAY8"

        if( pf == SinkFormats.BGRA ):
            return "BGRx"


class ImageCallbackData:
    ''' class for user data passed to the on new image callback function
    Params:
    newImageReceived    : True if a new image was received 
    image               : Last image received
    busy                : True if handler is busy  
    '''
    
    def __init__(self, newImageReceived, image, fps, img_h, img_w):
        self.newImageReceived = newImageReceived
        self.image = image
        self.fps = float(fps.split("/")[0])
        self.frame_height = img_h
        self.frame_width = img_w
        self.busy = False
        self.timestamp = None
        

def on_new_image(camera, userdata):
    '''
    Callback function, which will be called by the TIS class
    :param tis: the camera TIS class, that calls this callback
    :param userdata: This is a class with user data, filled by this call.
    :return:
    '''
    # Avoid being called, while the callback is busy
    if userdata.busy is True:
            return

    userdata.busy = True
    userdata.newImageReceived = True
    userdata.image = camera.Get_image()
    userdata.timestamp = datetime.datetime.now()
    userdata.busy = False

class CameraInterface:
    def __init__(self,  camera_type : CameraTypes, 
                        camera_connection,
                        frame_rate = "15/1",
                        frame_width = 640,
                        frame_height = 480, 
                        sink_format: SinkFormats = SinkFormats.BGRA):
        ''' Constructor
        :return: none
        '''
        self.camera_type = camera_type
        self.camera_connection = camera_connection

        self.frame_height = frame_height 
        self.frame_width = frame_width
        self.frame_rate = frame_rate
        self.sink_format = sink_format

        self.callback_data = ImageCallbackData( newImageReceived=False, 
                                                image=None,
                                                fps=self.frame_rate, 
                                                img_h=self.frame_height, 
                                                img_w=self.frame_width)
        self.image_handling = None

        print(self.camera_type.toString())

        self.setup_camera()
    
    def set_image_handling(self, function):
        self.image_handling = function

    def setup_camera(self):
        print(self.callback_data.newImageReceived)
        if self.camera_type == CameraTypes.TIS:
            self.camera = TIS.TIS()
            self.camera.openDevice( serial = self.camera_connection, 
                                    width = self.frame_width,
                                    height = self.frame_height,
                                    framerate = self.frame_rate,
                                    sinkformat = self.sink_format, 
                                    showvideo = False)
            self.camera.Set_Image_Callback(on_new_image, self.callback_data)

        elif self.camera_type == CameraTypes.CSI:
            self.camera = CSI_IP.CSI_IP(self.camera_type)
            self.camera.openDevice( connection = self.camera_connection, 
                                    width = self.frame_width,
                                    height = self.frame_height,
                                    framerate = self.frame_rate)
            self.camera.Set_Image_Callback(on_new_image, self.callback_data)
        elif self.camera_type == CameraTypes.IP:
            pass

        else:
            return False

    def start_pipeline(self):

        self.callback_data.busy = True 
        self.camera.Start_pipeline()
        self.callback_data.busy = False  

        if not self.image_handling is None:
            self.image_handling(self.callback_data)
        else:
            print("No image handling implemented.")
            self.stop_pipeline()

    def stop_pipeline(self):        
        self.camera.Stop_pipeline() 

