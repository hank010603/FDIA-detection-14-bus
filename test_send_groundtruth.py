# test_send_groundtruth.py
import socket, json, time
HOST = "127.0.0.1"
PORT = 5006

packet = {
    "data": {
        "V": [1.0]*14,
        "I": [0.0]*14
    },
    "metadata": {
        "ground_truth": 1,
        "note": "unit test - should be collected by udp_send"
    }
}

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(json.dumps(packet).encode(), (HOST, PORT))
print("Sent test packet with ground_truth=1 to 127.0.0.1:5006")
s.close()
time.sleep(0.1)
