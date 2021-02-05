import sys, os
sys.path.append(os.path.join(os.path.basename(__file__), "..", "src"))

from mqtt_connection import MQTTConnection

def test_send_test_message():
    conn = MQTTConnection()
    assert conn.sendTestMessage()
