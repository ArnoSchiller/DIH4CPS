from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='dih4cps',
    version='0.0.1',
    description='Including AI underwater shrimp detection with IoT connection for smart aquacultures.',
    py_modules=[
        "configuration", 
        "model_handler",
        "mqtt_connection", 
        "ring_buffer",
        "trained_model", 
        "video_processing",
        "webcam_capture",
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
        "tensorflow ~= 1.15",
        "paho-mqtt ~= 1.4",
        "opencv-python ~= 3.4",
        "boto3 ~= 1.10",
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