import numpy as np
import os, sys
import cv2
import tensorflow as tf
from datetime import datetime, timedelta

from collections import defaultdict
from io import StringIO

from model_handler import ModelHandler
from configuration import global_with_video_display

# BASE_PATH = os.path.abspath("C:/Users/Schiller/Google Drive/ObjectDetection")
#BASE_PATH = os.path.dirname(__file__)
#sys.path.append(os.path.join(BASE_PATH, "tf2_object_detection_API/models/research"))

from utils import label_map_util
if global_with_video_display:
    from utils import visualization_utils as vis_util

import subprocess


class Model:
    """
    Model: This class uses trained models to detect objects on images.
  
    Make sure u created a folder named as the model version and added every important files: 
        - model graph to a directory called created_model_graph. 
        - label map called object-detection.pbtxt to folder data

    @authors:   Arno Schiller (AS)
    @email:     schiller@swms.de
    @version:   v1.0.1
    @license:   see  https://github.com/ArnoSchiller/DIH4CPS-PYTESTS

    VERSION HISTORY
    Version:    (Author) Description:                               Date:
    v0.0.1      (AS) First initialize. Added first trained model    14.10.2020\n
                    called tf_API_data2_v1.
    v0.0.2      (AS) Included function to save frames if a shrimp   26.10.2020\n
                    was detected.
    v1.0.0      (AS) Included to v3.                                05.11.2020\n
    v1.0.1      (AS) Included streaming for detection results.      19.11.2020\n
    """

    #model_label_map_name    = "model_label_map.pbtxt"
    #model_graph_name        = "created_model_graph"

    #num_classes = 1
    #min_score_thresh = 0.5

    #image_dir = "images_detected"

    stream_results = False
    
    min_score_thresh = 0.8

    def __init__(self,  save_detected_frames=True, 
                        model_name=None, 
                        with_visualisation=global_with_video_display,
                        capture_params=None):

        MODEL_NAME = 'tf_API_data1_v01'
        MODEL_GRAPH = MODEL_NAME + '_graph_1' 
        
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        file_dir_path = os.path.dirname(__file__)
        PATH_TO_CKPT = os.path.join(file_dir_path, MODEL_GRAPH, 'frozen_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        # DATASET_PATH = os.path.join(BASE_PATH, "datasets")
        # DATASET_NAME = "dataset-v1-dih4cps"
        # PATH_TO_LABELS = os.path.join(DATASET_PATH, DATASET_NAME, 'labelmap.pbtxt')

        # DATASET_PATH = os.path.join(BASE_PATH, "datasets")
        DATASET_NAME = "dataset-v1-dih4cps"
       # PATH_TO_LABELS = os.path.join(DATASET_PATH, DATASET_NAME, 'labelmap.pbtxt')
        PATH_TO_LABELS = os.path.join(file_dir_path, 'model_label_map.pbtxt')

        NUM_CLASSES = 1 

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

    def calculate_box_intense(self, image, box):
        threshold_value = 128
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x_1 = int(box[0] * gray_image.shape[0])
        x_2 = int(box[2] * gray_image.shape[0])
        y_1 = int(box[1] * gray_image.shape[1])
        y_2 = int(box[3] * gray_image.shape[1])

        box_image = image[x_1:x_2, y_1:y_2, :]
        box_image_gray = gray_image[x_1:x_2, y_1:y_2]

        ret, box_image_gray = cv2.threshold(box_image_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        max_value = max(box_image_gray.flatten())
        min_value = min(box_image_gray.flatten())
        
        num_intense_px = np.count_nonzero(box_image_gray.flatten() > 0)

        res_dict = {}
        res_dict['max_value'] = max_value
        res_dict['min_value'] = min_value
        res_dict['threshold_value'] = threshold_value
        res_dict['num_intense_px'] = num_intense_px

        return res_dict

    def predict(self, frame):
        res_num_detected = 0
        res_boundingBoxes = [] 
        res_box_infos = []
        res_scores = []

        image_np = frame

        #ts = datetime.now()
        #print("TS: ", datetime.now() - ts)

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
                    [ boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                boxes = np.squeeze(boxes)

                """              
                if self.stream_results:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    # write to pipe
                    self.p.stdin.write(frame.tobytes())
                """

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                for index in range(boxes.shape[0]):
                    if scores[index] > self.min_score_thresh:
                        box = tuple(boxes[index].tolist())  

                        res_num_detected += 1
                        res_boundingBoxes.append(box)
                        res_scores.append(scores[index])
                        box_info = self.calculate_box_intense(image_np, boxes[index])
                        box_info_str = "{" 
                        for index, box_key in enumerate(box_info.keys()):
                            box_info_str += str(box_key) + "="
                            box_info_str += str(box_info[box_key])
                            if index < len(box_info.keys()) - 1:
                                box_info_str += "," 
                        box_info_str += "}" 
                        res_box_infos.append(box_info_str)

        return  res_num_detected, res_boundingBoxes, res_scores, res_box_infos

if __name__ == "__main__":
    print("Run test_trained_model")