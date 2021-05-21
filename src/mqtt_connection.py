"""
MQTTConnection: This class rules the connection to the MQTT broker (iotstack).

Requirements:
- paho (mqtt api): pip install paho-mqtt (see https://pypi.org/project/paho-mqtt/) Version 1.4.0

@authors:   Arno Schiller (AS)
@email:     schiller@swms.de
@version:   v4.0.0
@license:   https://github.com/ArnoSchiller/DIH4CPS-PYTESTS

VERSION HISTORY
Version:    (Author) Description:                                   Date:
vX.0.1           See v1-v3 (mqtt_connection) for more.	            XX-XX-XXXX\n
v4.0.0      (AS) Removed not needed functions. Updated function to  05-02-2021\n
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
    
    mqtt_host   = global_mqtt_host
    user_name   = global_mqtt_user_name
    password    = global_mqtt_password

    port        = global_mqtt_port
    keepalive   = 60

    local_mqtt  = global_mqtt_usinglocalhost

    topic = "IOT/test"


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
        msg = "testing MQTT"
        res = self.sendMessage(msg)
        return res[0] == 0

    def sendDetectionMessage(self,  
                            # context informations (tag_set)
                            user, 
                            location_ref,

                            # detection informations (tag_set)
                            model_name,
                            score_min_thresh,
                            iou_min_thresh,
                            detection_idx,
                            
                            # detection informations (field_set)
                            x_center,
                            y_center,
                            box_w,
                            box_h,
                            box_area,
                            detected_class_idx,
                            detected_score,

                            # additional tags (tag_set)
                            file_name=None, # for processing video files

                            # timestamp
                            timestamp=None):

        """
        Sending the detected bounding box information via MQTT.

        Influx string syntax like:
        shrimp,user={u},process=ProcessPreviousVideos,location={l},modelName={m},scoreMinThresh={s},iouMinThresh={i},detection_idx={idx}[,filename={f}] x_center={x_c},y_center={y_c},box_w={box_w},box_h={box_h},box_area={box_area},class_idx={class_idx},score={score} timestamp
        """

        # measurement
        msg = global_measurement
        
        # tag_set
        msg += ",user={}".format(user)
        msg += ",process=ProcessVideoStream"#ProcessPreviousVideos"
        msg += ",location={}".format(location_ref)
        msg += ",modelName={}".format(model_name)
        msg += ",scoreMinThresh={}".format(score_min_thresh)
        msg += ",iouMinThresh={}".format(iou_min_thresh)
        msg += ",detection_idx={}".format(detection_idx)

        if not file_name is None:
            msg += ",filename={}".format(file_name)

        # field_set
        msg += " "
        msg += "x_center={}".format(x_center)
        msg += ",y_center={}".format(y_center)
        msg += ",box_w={}".format(box_w)
        msg += ",box_h={}".format(box_h)
        msg += ",box_area={}".format(box_area)
        msg += ",class_idx={}".format(detected_class_idx)
        msg += ",score={}".format(detected_score)
        
        # timestamp
        if not timestamp is None:
            msg += " "
            msg += self.get_influx_timestamp(ts=timestamp)
        
        res = self.sendMessage(msg)
        return res
    
    def get_influx_timestamp(self, ts):
        """
        Converts a timestamp with format YEAR-MONTH-DAY_HOUR-MIN-SEC-MILLISEC to a nanosecond format (Difference between timestamp and basic timestamp 1677-09-21T00:12:43.145224194Z).

        Inputs:
            ts (int):       timestamp in ns format
            ts (string):    timestamp in YYYY-MM-DD_HH-MM-SS-MS format
            ts (datetime):  datetime object
        """
        
        if ts.__class__.__name__ == 'str':
            [ts_date, ts_time] = ts.split("_")
            ts_date = ts_date.split("-")
            ts_time = ts_time.split("-")

            ts = datetime.datetime(int(ts_date[0]), int(ts_date[1]),int(ts_date[2]), int(ts_time[0]),int(ts_time[1]),int(ts_time[2]),int(ts_time[3])*1000)

        if ts.__class__.__name__ == 'int':
            # --> allready in ns format
            return "{:.0f}".format(ts)

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

    #""" debugging MQTT
    conn = MQTTConnection()
    conn.testloop()
    #"""
