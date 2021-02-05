from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='dih4cps',
    version='4.0.1',
    description='Including AI underwater shrimp detection with IoT connection for smart aquacultures using pytorch YOLOv5 model.',
    
    scripts=['bin/dih4cps'],

    py_modules=[
        "configuration", 
        "mqtt_connection", 
        "detect",
        "video_capture",
    ],
    package_dir={'': 'src'},

    classifiers=[
        "Programming Language :: Python ::3",
        "Programming Language :: Python ::3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    long_description=long_description,
    long_description_content_type="text/markdown",

    install_requires = [
        "paho-mqtt ~= 1.4",
        "Cython",
        "matplotlib>=3.2.2",
        "numpy>=1.18.5",
        "opencv-python>=4.1.2",
        "Pillow",
        "PyYAML>=5.3.1",
        "scipy>=1.4.1",
        "tensorboard>=2.2",
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "tqdm>=4.41.0",
        "thop",  # FLOPS computation
        "pycocotools>=2.0",  # COCO mAP
    ],


    extras_reqire = {
        "dev": [
            "pytest>=3.7",
        ],
    },

    url = "https://github.com/ArnoSchiller/DIH4CPS-PYTESTS",
    author = "Arno Schiller",
    author_email = "schiller@swms.de",
)