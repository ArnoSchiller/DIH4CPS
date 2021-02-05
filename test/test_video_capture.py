import sys, os
sys.path.append(os.path.join(os.path.basename(__file__), "..", "src"))

from video_capture import VideoCapture

def test_capture_frame():
    vc = VideoCapture(True)
    assert vc.test_capture()
