import threading
import datetime, time 

from model_handler import ModelHandler
from mqtt_connection import MQTTConnection
from ring_buffer import RingBuffer
from trained_model import Model
from webcam_capture import WebcamCapture

from configuration import global_user_name

class VideoProcessor:
    """
    This class rules the video processing on video stram. 

    """
    user = global_user_name

    def __init__(self):
        """
        Create a buffer object and setup the model. After that start the webcam capture and video processing.
        """

        ## setup ring buffer 
        self.ring_buffer = RingBuffer()

        ## setup MQTT Connection
        self.mqtt_connection = MQTTConnection()

        ## define model name 
        self.used_model_name = "tf_API_data1_v01"

        ## download and validate the model 
        self.model_handler = ModelHandler()
        self.model_handler.download_trained_model(model_name=self.used_model_name)

        counter_try = 0
        while not self.model_handler.validate_model():
            print("Can not download model. Trying tu use default model.")
            counter_try += 1
            if counter_try >= 5:
                print("Error downloading model. Quitting...")
                quit()
            self.model_handler.download_trained_model(model_name=self.used_model_name)

        ## setup the model 
        self.model = Model()

        ## setup and start the video capture 
        self.video_capture_object = WebcamCapture(self.ring_buffer)
        self.video_capture_thread = threading.Thread(target=
                            self.video_capture_object.capture_frames)
        self.video_capture_thread.start()

        ## start the analyse thread 
        #self.video_processor_thread = threading.Thread(target= self.analyse_stream)
        #self.video_processor_thread.start()
        self.analyse_stream()

    def analyse_stream(self, buffer=None):

        if buffer is None:
            buffer = self.ring_buffer

        self.isRunnnig = True
        while self.isRunnnig:

            frame = buffer.get_next_element()[1]
            time_stamp = datetime.datetime.now()

            start_time = time.time()
            pred = self.model.predict(frame)
            pred_time = time.time() - start_time
            self.send_pred_mqtt(pred, time_stamp)
            send_time = time.time() - pred_time

            print("Pred: {}, Send: {}".format(pred_time, send_time))

    def send_pred_mqtt(self, prediction, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        [num_shrimps, boxes, scores] = prediction

        boxes_str = str(boxes)
        scores_str = str(scores)

        self.mqtt_connection.sendDetectionMessage( user=self.user,
                                process_version="01",
                                model_name=self.used_model_name,
                                score_min_thresh=self.model.min_score_thresh,
                                #frame_timestamp=timestamp,
                                boxes=boxes_str,
                                scores=scores_str,
                                num_shrimps=num_shrimps)

if __name__ == "__main__":
    VideoProcessor()