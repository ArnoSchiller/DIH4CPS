from mqtt_connection import MQTTConnection

def test_send_test_message():
    conn = MQTTConnection()
    assert conn.sendTestMessage()
