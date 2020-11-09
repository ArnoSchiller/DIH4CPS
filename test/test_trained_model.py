import cv2
import os

from model_handler import ModelHandler
from trained_model import Model

def test_download_trained_model():
    mh = ModelHandler()
    assert mh.download_trained_model()

def test_validate_model():
    mh = ModelHandler()
    assert mh.validate_model()
    
def test_predict():
    mh = ModelHandler()
    path = os.path.dirname(__file__)
    if mh.download_test_image(base_path=path):
        file_path = os.path.join(path, mh.test_image_name)
        image = cv2.imread(file_path)
        model = Model()
        res = model.predict(image)
        os.remove(file_path)
        if len(res) >= 3:
            assert True
        else:
            assert False
    else:
        assert False

