from configuration import global_location, global_user_name
from camera_interface import ImageCallbackData

import datetime
import numpy as np

def handle_detection(data:ImageCallbackData):
    
    detector = Yolov5Detection(data.fps, data.frame_width, data.frame_height)
    
    model_name = "yolov5l_dsv1_aug"
    detector.setup_model(model_name=model_name)
    
    if data.newImageReceived:
        img = data.image.copy()
        # Padded resize
        t0 = time.time()
        detector.run_detection_on_image(img, data.timestamp)
        if detector.save_txt or detector.save_img:
            s = f"\n{len(list(detector.save_dir.glob('labels/*.txt')))} labels saved to {detector.save_dir / 'labels'}" if detector.save_txt else ''
            print(f"Results saved to {detector.save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')



""" include packages for object detection """

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class Yolov5Detection:
        
    mode = None
    model_name = None     # model name (set by init)
    model_weights = None  # locations of model weights (set by automatically) 
                            # weights/'model_name'/model_weights.pt
    img_size = None       # image size (set by init)
    stride = 32

    project_dir = 'runs/detect'
    weigths_dir = 'weights'
    save_dir = None
    exist_ok = True       # existing project/name ok, do not increment

    conf_thres = None     # object confidence threshold (set by init)
    iou_thres = None      # IOU threshold for NMS (set by init)
    classes = None        # filter by class: --class 0, or --class 0 2 3'
    agnostic_nms = False  # set True for class-agnostic NMS

    save_txt = None       # save results as txt if true (set by init)
    send_mqtt = None      # send results via MQTT (set by init)
    save_img = None
    view_img = None
    save_conf = None

    augment = False       # augmented inference

    names = None          # filtered class names
    colors = None         # one color for each class
    device = None
    half = None
    model = None
    mqtt = None

    fps = None
    frame_width = None
    frame_height = None

    def __init__(self, fps=10, frame_w=640, frame_h=480):
        self.fps = fps
        self.frame_width = frame_w 
        self.frame_height = frame_h

    def setup_model(self, model_name,
                    mode="video",
                    img_size = 640,
                    conf_thres = 0.5,
                    iou_thres = 0.5, 
                    save_txt = False,
                    send_mqtt = True,
                    save_img = False,
                    view_img = True,
                    save_conf = True):

        # update parameters
        self.mode = mode
        self.model_name = model_name
        self.model_weights = Path(self.weigths_dir , self.model_name , "model_weights.pt")
        print(f"Using weights: {self.model_weights}")

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.save_txt = save_txt
        self.send_mqtt = send_mqtt
        self.save_img = save_img
        self.view_img = view_img
        self.save_conf = save_conf 

        # used parameters
        device = ''

        # Directories
        self.save_dir = Path(increment_path(Path(self.project_dir) / self.model_name, exist_ok=self.exist_ok))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # setup MQTT connection 
        if self.send_mqtt:
            from mqtt_connection import MQTTConnection
            self.mqtt = MQTTConnection()

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.model_weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def run_detection_on_image(self, im0, timestamp=""):

        img = letterbox(im0, self.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ### Addition: timestamp in ns
        #timestamp_ns = time.time_ns() # python >= 3.7
        timestamp_ns = int(time.time() * 1000000000)
        ##########################################
            
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
                       
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                s = timestamp_to_string(timestamp)
                out_name = s
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                detection_idx = 0
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        txt_path = str(self.save_dir / s )
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.view_img:  # Add bbox to image
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

                    ### Addition: sending results via mqtt
                    if self.send_mqtt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        self.mqtt.sendDetectionMessage(user=global_user_name,
                                location_ref= global_location, 
                                model_name=self.model_name,
                                score_min_thresh=self.conf_thres,
                                iou_min_thresh=self.iou_thres,
                                detection_idx=detection_idx,
                                x_center=xywh[0],
                                y_center=xywh[1],
                                box_w=xywh[2],
                                box_h=xywh[3],
                                box_area= xywh[2] * xywh[3],
                                detected_class_idx=int(cls),
                                detected_score=conf,
                                timestamp=int(timestamp_ns))
                    detection_idx += 1
                    ##########################################

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if self.view_img:
                cv2.imshow(str(out_name), im0)
                cv2.waitKey(1)
            save_path = str(self.save_dir / out_name)
            # Save results (image with detections)
            if self.save_img:
                if self.mode == 'image':
                    cv2.imwrite(save_path+".png", im0)
                else:  # 'video'
                    if self.vid_path != save_path:  # new video
                        self.vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = self.fps
                        w = int(self.frame_width)
                        h = int(self.frame_height)
                        vid_writer = cv2.VideoWriter(save_path+".avi", cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def timestamp_to_string(ts=None):
    """ function to format datetime timestamp like 
    YEAR-MONTH-DAY_HOUR-MINUTE-SECOND
    """
    if ts is None:
        ts = datetime.datetime.now()
    return f"{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"