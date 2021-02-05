import cv2
import time 

from configuration import global_camera_connection

class VideoCapture:
    
    def __init__(self, testing=False):
        self.capture = cv2.VideoCapture(global_camera_connection)
        self.testing = testing

    def test_capture(self):
        if not self.capture.isOpened():
            return False 

        ret, frame = self.capture.read()
        if not frame is None:
            if not self.testing:
                cv2.imshow("Frame", frame)
            return True

        return False
    
    def testloop(self, duration_s = 10):
        t1 = time.time()
        while time.time() - t1 < duration_s:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
