from webcam_capture import WebcamCapture

def test_init_video_capture():
    assert not WebcamCapture(None) == None
    
def test_connect_to_camera():
    cap = WebcamCapture(None)
    assert cap.reconnect(testing=True)

def test_capture_test_frame():
    cap = WebcamCapture(None)
    assert cap.capture_test_frame()
