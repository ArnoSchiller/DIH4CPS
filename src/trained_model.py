import numpy as np
import os, sys
import cv2
import tensorflow as tf
from datetime import datetime, timedelta

from collections import defaultdict
from io import StringIO
file_dir_path = os.path.dirname(__file__)

from model_handler import ModelHandler
from configuration import global_with_video_display

if not os.path.exists(os.path.join(file_dir_path, "utils")):
    print("could not find utils.")
    mh = ModelHandler()
    mh.download_utils(with_visualisation=global_with_video_display)


from utils import label_map_util
if global_with_video_display:
    from utils import visualization_utils as vis_util


class Model:
    """
    Model: This class uses trained models to detect objects on images.
  
    Make sure u created a folder named as the model version and added every important files: 
        - model graph to a directory called created_model_graph. 
        - label map called object-detection.pbtxt to folder data

    @authors:   Arno Schiller (AS)
    @email:     schiller@swms.de
    @version:   v0.0.2
    @license:   see  https://github.com/ArnoSchiller/DIH4CPS-PYTESTS

    VERSION HISTORY
    Version:    (Author) Description:                               Date:
    v0.0.1      (AS) First initialize. Added first trained model    14.10.2020\n
                    called tf_API_data2_v1.
    v0.0.2      (AS) Included function to save frames if a shrimp   26.10.2020\n
                    was detected.
    v1.0.0      (AS) Included to v3.                                05.11.2020\n
                    was detected.
    """
    model_label_map_name    = "model_label_map.pbtxt"
    model_graph_name        = "created_model_graph"

    num_classes = 1
    min_score_thresh = 0.5

    image_dir = "images_detected"

    def __init__(self,  save_detected_frames=True, 
                        model_name=None, 
                        with_visualisation=global_with_video_display):

        # num classes anpassen

        self.save_detected_frames = save_detected_frames
        self.with_visualisation = with_visualisation
        self.model_name = model_name
        
        self.model_handler = ModelHandler()
        if not self.model_handler.validate_model():
            self.model_handler.download_trained_model(model_name)

        # proof if label map is reachable
        self.model_label_map_path = self.model_handler.label_map_path
        print(self.model_label_map_path)
        if not os.path.exists(self.model_label_map_path):
            print("could not find label map.")
            quit()

        # proof if model graph is reachable
        model_graph_path =self.model_handler.model_graph_path
        if not os.path.exists(model_graph_path):
            print("could not find model graph.")
            quit()
        self.model_ckpt_path = os.path.join(model_graph_path, 'frozen_inference_graph.pb')

        ## load the (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        ## load the label map
        label_map = label_map_util.load_labelmap(self.model_label_map_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        if self.save_detected_frames:
            if not os.path.exists(self.image_dir):
                os.mkdir(self.image_dir)


    def predict(self, frame):
        image_np = frame
        res_num_detected = 0
        res_boundingBoxes = [] 
        res_scores = []

        with self.detection_graph.as_default():
            with tf.compat.v1.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                
                # Visualization of the results of a detection.
                """ not needed for edge processing
                if self.with_visualisation:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
            
                    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()  
                """
                
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                for i in range(boxes.shape[0]):
                    if scores[i] > self.min_score_thresh:
                        box = tuple(boxes[i].tolist())  

                        res_num_detected += 1
                        res_boundingBoxes.append(box)
                        res_scores.append(scores[i])

        return  res_num_detected, res_boundingBoxes, res_scores

    def analyse_video(self, file_path): 

        video_capture = cv2.VideoCapture(file_path)

        time_stamps = []
        num_shrimps = []
        boundingBoxes = []
        scores = []

        video_name = os.path.basename(file_path).split(".")[0]

        video_time = video_name.split("_")[-1].split("-")
        video_date = video_name.split("_")[-2].split("-")

        timestamp = datetime(   int(video_date[0]), 
                                int(video_date[1]), 
                                int(video_date[2]), 
                                int(video_time[0]),
                                int(video_time[1]),
                                int(video_time[2]),
                                0)

        counter = 0 
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            output_frame = np.copy(frame)
            counter += 1
            if ret:
                
                timestamp_str = "{0}-{1:02d}-{2:02d}_{3:02d}-{4:02d}-{5:02d}-{6:03d}".format(timestamp.year, 
                            timestamp.month,
                            timestamp.day,
                            timestamp.hour,
                            timestamp.minute,
                            timestamp.second,
                            int(timestamp.microsecond/1000))
                timestamp = timestamp + timedelta(microseconds=self.delta_ms * 1000)
                
                frame_name = video_name + "_{}".format(counter)
                frame_shrimps, frame_bb, frame_scores = self.predict(frame)

                time_stamps.append(timestamp_str)
                num_shrimps.append(frame_shrimps)
                boundingBoxes.append(np.squeeze(frame_bb).tolist())
                scores.append(frame_scores)

                if self.save_detected_frames and frame_shrimps > 0:
                    cv2.imwrite(os.path.join(self.image_dir, frame_name + ".png"), output_frame)

            else:
                break

        return [time_stamps, num_shrimps, boundingBoxes, scores]

if __name__ == "__main__":
    print("Run test_trained_model")