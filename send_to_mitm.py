import socket
import json
import time
import numpy as np

SIMULINK_PORT = 5005
SEND_INTERVAL = 0.05
NUM_BUSES = 42  # 改成 42 buses

def generate_measurements():
    V = np.random.uniform(0.95, 1.05, NUM_BUSES).tolist()  # 模擬電壓
    I = np.random.uniform(-0.05, 0.05, NUM_BUSES).tolist()  # 模擬電流
    return {'V': V, 'I': I}

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[+] Sending simulated 42-bus measurements to UDP port {SIMULINK_PORT} ...")
    
    try:
        while True:
            data = generate_measurements()
            sock.sendto(json.dumps(data).encode(), ('127.0.0.1', SIMULINK_PORT))
            time.sleep(SEND_INTERVAL)
    except KeyboardInterrupt:
        print("\n[!] Stopped sending measurements.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()

