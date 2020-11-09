"""
MQTTConnection: This class rules the connection to the MQTT broker (iotstack).

Requirements:
- paho (mqtt api): pip install paho-mqtt (see https://pypi.org/project/paho-mqtt/) 
  Version 1.4.0

@authors:   Arno Schiller (AS)
@email:     schiller@swms.de
@version:   v2.0.0
@license:   https://github.com/ArnoSchiller/DIH4CPS-PYTESTS

VERSION HISTORY
Version:    (Author) Description:                                   Date:
v0.0.1           See v1 (mqtt_connection) for more.    	            07-08-2020\n
v1.0.0      (AS) Updated to version 2.                              07-09-2020\n
v1.0.1      (AS) Updated code in case there is no internet          07-09-2020\n
                connction by using try and except.                            \n
v1.1.0      (AS) Included functions to send shrimp detections via   21-10-2020\n
                MQTT. Sending the number of shrimps for each frame.           \n
v2.0.0      (AS) Included into v3, added more documentation.        04-11-2020\n
v2.0.1      (AS) Created test function.                             04-11-2020\n
"""

import paho.mqtt.client as mqtt
import datetime, time
from configuration import *

class MQTTConnection:
    """
    This class rules the connection to the MQTT broker (iotstack).

    Attributes:
    -----------
    mqtt_host : str
        the hostname or IP address of the remote broker
    user_name : str
        username of the MQTT client
    password : str
        password of the MQTT client
    port : int
        the network port of the server host to connect to. Defaults to 1883. 
        Note that the default port for MQTT over SSL/TLS is 8883 so if you are 
        using tls_set() or tls_set_context(), the port may need providing manually
    keepalive : int
        maximum period in seconds allowed between communications with the broker. 
        If no other messages are being exchanged, this controls the rate at which 
        the client will send ping messages to the broker

    local_mqtt : bool
        True    - connect to local MQTT
        False   - connect to mqtt_host

    status_list : dict
        list of possible informations, warnings and errors to send to MQTT server 
        (API for WebcamCapture and CloudConnection).
    """
    # SSL/TLS1.2 aktivieren. Pub auf den Topic IOT/{irgendwas}
    
    mqtt_host   = global_mqtt_host
    user_name   = global_mqtt_user_name
    password    = global_mqtt_password

    port        = global_mqtt_port
    keepalive   = 60

    local_mqtt  = global_mqtt_usinglocalhost

    topic = "IOT/test"

    status_list = {
        "WebcamCapture" : {
            # Open camera
            "ClosedCamera"          : "modul=WebcamCapture,process=OpenCamera status=0",
            "OpeningCamera"         : "modul=WebcamCapture,process=OpenCamera status=1",
            "OpenedCamera"          : "modul=WebcamCapture,process=OpenCamera status=2",

            "OpeningCameraFailed"   : "modul=WebcamCapture,process=OpenCamera status=4",

            "OpenCameraError"       : "modul=WebcamCapture,process=OpenCamera status=10",

            "CapturedFrame"         : "modul=WebcamCapture,process=CaptureFrame status=2",
        },

        "VideoRecorder" : {
            # Open writer
            "ClosedWriter"          : "modul=VideoRecorder,process=OpenWriter status=0",
            "OpeningWriter"         : "modul=VideoRecorder,process=OpenWriter status=1",
            "OpenedWriter"          : "modul=VideoRecorder,process=OpenWriter status=2",

            "OpenFileWriterFailed"  : "modul=VideoRecorder,process=OpenWriter status=4",

            "OpenWriterError"       : "modul=VideoRecorder,process=OpenWriter status=10",

            # Recording file
            "RecordingFile"         : "modul=VideoRecorder,process=RecordFile status=1",
            "RecordedFile"          : "modul=VideoRecorder,process=RecordFile status=2",

            "RecordLostConnection"  : "modul=VideoRecorder,process=RecordFile status=9",
            "RecordFileError"       : "modul=VideoRecorder,process=RecordFile status=10",
        },

        "CloudConnection" : {
            # Upload files 
            "UploadReady"       	: "modul=CloudConnection,process=UploadFile status=0",
            "UploadingFile"         : "modul=CloudConnection,process=UploadFile status=1",
            "UploadedFile"          : "modul=CloudConnection,process=UploadFile status=2",

            "ClientError"           : "modul=CloudConnection,process=UploadFile status=7",
            "FileNotFound"          : "modul=CloudConnection,process=UploadFile status=8",
            "NoCredentials"         : "modul=CloudConnection,process=UploadFile status=9",
            "UploadError"           : "modul=CloudConnection,process=UploadFile status=10",

            # connecting to server
            "DisconnectedServer"    : "modul=CloudConnection,process=ConnectServer status=0",
            "ConnectingToServer"    : "modul=CloudConnection,process=ConnectServer status=1",
            "ConnectedToServer"     : "modul=CloudConnection,process=ConnectServer status=2",

            "ConnectServerError"    : "modul=CloudConnection,process=ConnectServer status=10"
        }
    }

    def __init__(self):
        """ 
        Setup and configure the MQTT connection. 
        """
        self.client = mqtt.Client()
        self.client._username = self.user_name
        self.client._password = self.password
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.client.tls_set()

        self.reconnect()
    

    def reconnect(self):
        """
        (re-)connect to the MQTT broker. 
        """
        if self.local_mqtt:
            self.client.connect("localhost", 1883, 60)
        else: 
            try:
                self.client.connect(self.mqtt_host, self.port, self.keepalive)
            except Exception:
                pass
        self.client.loop_start()

    def testloop(self):
        """
        Send test messages to the broker every 5 seconds. 
        """
        while True:
            self.sendTestMessage()
            time.sleep(5)
    
    def sendTestMessage(self):
        """
        Send test messages to the broker. Returns true, if sending was successful, else return false.
        """
        msg = "test"
        res = self.sendMessage(msg)
        return res[0] == 0

    def sendProcessMessage(self, user, message, **options):
        """
        Send a process status message to the broker using the influx syntax. All possible messages are listed in status_list seperat for every process. Also you can add more options:
        --file      name of current file
        """
        msg = "process,user={},".format(user)
        if not options.get("file")  == None:
            msg += "file={},".format(options.get("file")) 
        msg += "{}".format(message)
        res = self.sendMessage(msg)

    def sendDetectionMessage(self,  user, 
                                    process_version, 
                                    model_name,
                                    score_min_thresh,
                                    num_shrimps,
                                    frame_timestamp=None,
                                    boxes=None,
                                    scores=None,
                                    process_timestamp=None,
                                    file_name=None):
        """
        Sending the detected number of shrimps via MQTT.

        Influx string syntax like:
        shrimp,user={u},modul=VideoProcessing,process=ProcessPreviousVideos, version={v},modelName={m}, scoreMinThresh={s}[, processTimestamp={p},filename={f},boxes={b},scores{c}] numShrimps={n} timestamp
        """
        msg = "shrimp,user={}".format(user)
        msg += ",modul=VideoProcessing,process=ProcessVideoStream"#ProcessPreviousVideos"
        msg += ",version={}".format(process_version)
        msg += ",modelName={}".format(model_name)
        msg += ",scoreMinThresh={}".format(score_min_thresh)
        if not process_timestamp is None:
            msg += ",processTimestamp={}".format(process_timestamp)
        if not file_name is None:
            msg += ",filename={}".format(file_name)
        if not boxes is None:
            msg += ",boxes={}".format(boxes)
        if not scores is None:
            msg += ",scores={}".format(scores)
        msg += " "
        msg += "numShrimps={}".format(num_shrimps)
        if not frame_timestamp is None:
            msg += " "
            msg += self.get_influx_timestamp(ts=frame_timestamp)
        print(msg)
        res = self.sendMessage(msg)

    def get_influx_timestamp(self, ts):
        """
        Converts a timestamp with format YEAR-MONTH-DAY_HOUR-MIN-SEC-MILLISEC to a millisecond format (Difference between timestamp and basic timestamp 1677-09-21T00:12:43.145224194Z).
        """
        
        if ts.__class__.__name__ == 'str':
            [ts_date, ts_time] = ts.split("_")
            ts_date = ts_date.split("-")
            ts_time = ts_time.split("-")

            ts = datetime.datetime(int(ts_date[0]), int(ts_date[1]),int(ts_date[2]), int(ts_time[0]),int(ts_time[1]),int(ts_time[2]),int(ts_time[3])*1000)

        # 1677-09-21T00:12:43.145224194Z
        base_ts = datetime.datetime(1677, 9, 21, 0, 12, 43, 145224)
        base_ns = -9223372036854775806

        delta_ts = ts - base_ts
        delta_ns = delta_ts.total_seconds()*1000000000
        
        ts_ns = base_ns + delta_ns

        return "{:.0f}".format(ts_ns)

    def sendMessage(self, message):
        """
        Publish message to broker.
        """
        return self.client.publish("IOT/test", message)
        

def on_connect(client, userdata, flags, rc):
    """ The callback for when the client receives a CONNACK response from the server.
    see getting started (https://pypi.org/project/paho-mqtt/)
    """
    # print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    #client.subscribe("IOT/#")

def on_message(client, userdata, msg):
    """ The callback for when a PUBLISH message is received from the server.
    see getting started (https://pypi.org/project/paho-mqtt/)
    """
    print(msg.topic+" "+str(msg.payload))

if __name__ == '__main__':
    print("Run test_mqtt_connection.")

    """ debugging Influx timestamp
    conn = MQTTConnection()
    print(conn.get_influx_timestamp("2020-10-19_20-30-45-250"))
    print(time.time_ns())
    #"""
    """ debugging MQTT
    conn = MQTTConnection()
    conn.testloop()
    conn.sendMessage("temperature,location=CPU,modul=SystemMonitoring,user=test1 temperature=20")
    #"""