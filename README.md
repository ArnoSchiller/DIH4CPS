# DIH4CPS
This package is used to test python packaging. 

## Installation 
```shell
pip install dih4cps
```

## Usage
Shell:
```shell
dih4cps
```

Python:
```python
cd src
python detect.py --source 701 --weights weights/yolov5l_dsv1_aug --send-mqtt 
```

##Testing with pytest
Make sure there is a test folder and run
```python
pytest
```
This will test every important functionality.
