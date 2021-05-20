from configuration import global_location, global_user_name
from camera_interface import ImageCallbackData

def handle_detection(*data:ImageCallbackData):
    setup_model()
    
    t0 = time.time()
    run_detection_on_image(data.image, data.timestamp)
    if m_save_txt or m_save_img:
        s = f"\n{len(list(m_save_dir.glob('labels/*.txt')))} labels saved to {m_save_dir / 'labels'}" if m_save_txt else ''
        print(f"Results saved to {m_save_dir}{s}")

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

m_mode = None
m_model_name = None     # model name (set by init)
m_model_weights = None  # locations of model weights (set by automatically) 
                        # weights/'model_name'/model_weights.pt
m_img_size = None       # image size (set by init)

m_project_dir = 'runs/detect'
m_weigths_dir = 'weights'
m_save_dir = None
m_exist_ok = True       # existing project/name ok, do not increment

m_conf_thres = None     # object confidence threshold (set by init)
m_iou_thres = None      # IOU threshold for NMS (set by init)
m_classes = ''          # filter by class: --class 0, or --class 0 2 3'
m_agnostic_nms = False  # set True for class-agnostic NMS

m_save_txt = None       # save results as txt if true (set by init)
m_send_mqtt = None      # send results via MQTT (set by init)
m_save_img = None
m_view_img = None
m_save_conf = None

m_augment = False       # augmented inference

m_names = None          # filtered class names
m_colors = None         # one color for each class
m_device = ''
m_half = None
m_model = None
m_mqtt = None


def setup_model(model_name,
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
    m_mode = mode
    m_model_name = model_name
    m_model_weights = Path(m_weigths_dir / m_model_name / "model_weights.pt")
    print(f"Using weights: {m_model_weights}")

    m_img_size = img_size
    m_conf_thres = conf_thres
    m_iou_thres = iou_thres
    m_save_txt = save_txt
    m_send_mqtt = send_mqtt
    m_save_img = save_img
    m_view_img = view_img
    m_save_conf = save_conf 

    # Directories
    m_save_dir = Path(increment_path(Path(m_project_dir) / m_model_name, exist_ok=m_exist_ok))  # increment run
    (m_save_dir / 'labels' if m_save_txt else m_save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # setup MQTT connection 
    if m_send_mqtt:
        from mqtt_connection import MQTTConnection
        m_mqtt = MQTTConnection()

    # Initialize
    set_logging()
    m_device = select_device(m_device)
    m_half = m_device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    m_model = attempt_load(m_model_weights, map_location=m_device)  # load FP32 model
    stride = int(m_model.stride.max())  # model stride
    imgsz = check_img_size(m_img_size, s=stride)  # check img_size
    if m_half:
        m_model.half()  # to FP16

    # Get names and colors
    m_names = m_model.module.names if hasattr(m_model, 'module') else m_model.names
    m_colors = [[random.randint(0, 255) for _ in range(3)] for _ in m_names]

    # Run inference
    if m_device.type != 'cpu':
        m_model(torch.zeros(1, 3, imgsz, imgsz).to(m_device).type_as(next(m_model.parameters())))  # run once

def run_detection_on_image(img, timestamp=""):

    img = torch.from_numpy(img).to(m_device)
    img = img.half() if m_half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    ### Addition: timestamp in ns
    #timestamp_ns = time.time_ns() # python >= 3.7
    timestamp_ns = int(time.time() * 1000000000)
    ##########################################
        
    # Inference
    t1 = time_synchronized()
    pred = m_model(img, augment=m_augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, m_conf_thres, m_iou_thres, classes=m_classes, agnostic=m_agnostic_nms)
    t2 = time_synchronized()

    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        """
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        """
        im0 = img.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            s = timestamp
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {m_names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            detection_idx = 0
            for *xyxy, conf, cls in reversed(det):
                if m_save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if m_save_conf else (cls, *xywh)  # label format
                    txt_path = str(m_save_dir / s )
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if m_save_img or m_view_img:  # Add bbox to image
                    label = f'{m_names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=m_colors[int(cls)], line_thickness=3)

                ### Addition: sending results via mqtt
                if m_send_mqtt:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    m_mqtt.sendDetectionMessage(user=global_user_name,
                            location_ref= global_location, 
                            model_name=m_model_name,
                            score_min_thresh=m_conf_thres,
                            iou_min_thresh=m_iou_thres,
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
        if m_view_img:
            cv2.imshow(str(s), im0)
        save_path = str(m_save_dir / s)
        # Save results (image with detections)
        if m_save_img:
            if m_mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)

