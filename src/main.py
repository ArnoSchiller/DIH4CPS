from pathlib import Path
import datetime
import argparse
import time
import cv2
import os

from cameras.camera_interface import CameraInterface, ImageCallbackData, CameraTypes
# from yolov5_detection import handle_detection

def live_stream(data:ImageCallbackData):

    error = 0
    print('Press Esc to stop')
    lastkey = 0
    cv2.namedWindow('Window',cv2.WINDOW_NORMAL)
    
    try:
        while lastkey != 27 and error < 5:

                while data.newImageReceived is False:
                    pass

                # Check, whether there is a new image and handle it.
                if data.newImageReceived is True:
                        print(data.timestamp)
                        data.newImageReceived = False
                        cv2.imshow('Window', data.image)
                else:
                        print("No image received")

                lastkey = cv2.waitKey(10)

    except KeyboardInterrupt:
        cv2.destroyWindow('Window')


def record_video(data:ImageCallbackData):
    
    # create dir 
    rec_dir = Path(Path("recordings") / "videos") 
    rec_dir.mkdir(parents=True, exist_ok=True)

    rec_file_path = Path(rec_dir, timestamp_to_string() + ".mp4")
 
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    rec_file = cv2.VideoWriter( str(rec_file_path), fourcc, data.fps, (data.frame_width,data.frame_height))
    
    if opt.show_stream:
        cv2.namedWindow('Recording...',cv2.WINDOW_NORMAL)
    
    lastkey = 0
    print('Press Esc or CTRL+C to stop recording')

    try:
        while lastkey != 27:

                while data.newImageReceived is False:
                    pass

                # Check, whether there is a new image and handle it.
                if data.newImageReceived is True:
                        print(data.timestamp)
                        data.newImageReceived = False
                        if opt.show_stream:
                            cv2.imshow('Recording...', data.image)
                        rec_file.write(data.image)
                else:
                        print("No image received")

                lastkey = cv2.waitKey(10)

    except KeyboardInterrupt:
        rec_file.release()
        cv2.destroyAllWindows()

def timestamp_to_string(ts=None):
    """ function to format datetime timestamp like 
    YEAR-MONTH-DAY_HOUR-MINUTE-SECOND
    """
    if ts is None:
        ts = datetime.datetime.now()
    return f"{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"

"""
handle_detection()
print('Press Esc to stop')

ci = CameraInterface(   camera_type=CameraTypes.TIS,
                        camera_connection="06120036",
                        frame_rate="10/1" )
ci.set_image_handling(show_image)
ci.start_pipeline()

ci.stop_pipeline()
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--function', type=str, default='live_stream', help='live_stream, record_video or run_detection')
    parser.add_argument('--show_stream', action='store_true', help='set true to view stream')
    opt = parser.parse_args()
    
    ci = CameraInterface(   camera_type=CameraTypes.TIS,
                            camera_connection="06120036",
                            frame_rate="10/1" )
    """
    ci = CameraInterface(   camera_type=CameraTypes.CSI,
                            camera_connection="1",
                            frame_rate="10/1" )
    #"""
    
    if opt.function == 'live_stream':
        ci.set_image_handling(live_stream)
    elif opt.function == 'record_video':
        ci.set_image_handling(record_video)
    else:
        print("Function not valid.")
        quit()

    ci.start_pipeline()