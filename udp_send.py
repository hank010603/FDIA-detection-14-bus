import socket
import asyncio
import websockets
import json
import time
import sys
import numpy as np
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class UDPServer:
    def __init__(self, udp_port: int = 5006, ws_port: int = 8765):
        self.udp_port = udp_port
        self.ws_port = ws_port
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind(('0.0.0.0', udp_port))

        # --- FDIA Model load ---
        warnings.filterwarnings("default", category=InconsistentVersionWarning)

        self.fdia_model = None
        self.preprocessor = None
        model_path = "enhanced_fdia_model_2_from_siddhartha.joblib"  # 改成實際檔名

        try:
            model_data = joblib.load(model_path)
            if hasattr(model_data, "predict"):
                self.fdia_model = model_data
                print(f"[+] Loaded model directly from {model_path}")
            elif isinstance(model_data, dict) and "model" in model_data:
                self.fdia_model = model_data["model"]
                self.preprocessor = model_data.get("preprocessor", None)
                print(f"[+] Loaded model+preprocessor from {model_path}")
            else:
                print(f"[!] Unknown model format in {model_path}: {type(model_data)}")
        except Exception as e:
            print(f"[!] Error loading model '{model_path}': {e}")

        print(f"[+] UDP server initialized on port {udp_port}")
        print(f"[+] WebSocket server will run on port {ws_port}")

        if sys.platform == 'win32':
            self.udp_sock.setblocking(False)

        # --- Metrics storage ---
        self.y_true = []
        self.y_pred = []

    async def detect_fdia(self, measurements):
        """FDIA detection using joblib model + optional preprocessor"""
        if not self.fdia_model or not measurements:
            return False, 0.0

        try:
            features = np.array(measurements).reshape(1, -1)

            if self.preprocessor is not None:
                try:
                    features = self.preprocessor.transform(features)
                except Exception as e:
                    print(f"[!] Preprocessor transform error: {e}")

            prediction = None
            try:
                prediction = self.fdia_model.predict(features)
            except Exception as e:
                print(f"[!] Model predict error: {e}")
                return False, 0.0

            probability = 0.0
            if hasattr(self.fdia_model, "predict_proba"):
                try:
                    probability = float(self.fdia_model.predict_proba(features)[0][1])
                except Exception as e:
                    print(f"[!] predict_proba error: {e}")
                    try:
                        probability = float(prediction[0])
                    except:
                        probability = 0.0
            else:
                try:
                    probability = float(prediction[0])
                except:
                    probability = 0.0

            return bool(prediction[0]), float(probability)

        except Exception as e:
            print(f"FDIA detection error: {str(e)}")
            return False, 0.0

    async def receive_udp(self):
        loop = asyncio.get_event_loop()
        if sys.platform == 'win32':
            while True:
                try:
                    data, addr = self.udp_sock.recvfrom(8192)
                    return data, addr
                except BlockingIOError:
                    await asyncio.sleep(0.01)
        else:
            return await loop.sock_recvfrom(self.udp_sock, 8192)

    async def try_parse_data(self, data):
        if all(byte == 0 for byte in data):
            print("[!] Received all null bytes - skipping")
            return None

        if len(data) < 5:
            print(f"[!] Data too short: {data}")
            return None

        try:
            decoded = data.decode('utf-8').strip()
            if not decoded:
                return None

            decoded = decoded.strip('\x00').strip()

            try:
                parsed = json.loads(decoded)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            try:
                start = decoded.find('{')
                end = decoded.rfind('}') + 1
                if start != -1 and end != -1 and start < end:
                    parsed = json.loads(decoded[start:end])
                    if isinstance(parsed, dict):
                        return parsed
            except json.JSONDecodeError:
                pass

            if '{' in decoded:
                try:
                    json_start = decoded.find('{')
                    possible_json = decoded[json_start:]
                    parsed = json.loads(possible_json)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

            return None
        except UnicodeDecodeError:
            for encoding in ['utf-16', 'latin-1', 'ascii']:
                try:
                    decoded = data.decode(encoding).strip().strip('\x00')
                    if decoded:
                        try:
                            return json.loads(decoded)
                        except json.JSONDecodeError:
                            continue
                except:
                    continue

            try:
                start_marker = b'{'
                end_marker = b'}'
                start_pos = data.find(start_marker)
                end_pos = data.rfind(end_marker)

                if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                    json_bytes = data[start_pos:end_pos + 1]
                    try:
                        return json.loads(json_bytes.decode('utf-8'))
                    except:
                        pass
            except:
                pass
        except Exception as e:
            print(f"Parse error: {e}")

        print(f"[!] Failed to parse data after all attempts: {data[:50]}...")
        return None

    async def validate_data_structure(self, parsed):
        if not isinstance(parsed, dict):
            return False

        if 'V' in parsed and 'I' in parsed:
            return True
        if 'data' in parsed and isinstance(parsed['data'], dict):
            if 'V' in parsed['data'] and 'I' in parsed['data']:
                return True
        return False

    async def extract_measurements(self, parsed):
        if 'V' in parsed and 'I' in parsed:
            return parsed['V'] + parsed['I']
        elif 'data' in parsed and isinstance(parsed['data'], dict):
            return parsed['data'].get('V', []) + parsed['data'].get('I', [])
        return []

    async def handle_websocket(self, websocket):
        print("[+] WebSocket client connected")
        try:
            while True:
                try:
                    data, addr = await self.receive_udp()
                    print(f"[+] Received {len(data)} bytes from {addr}")

                    if not data:
                        print("[-] Empty UDP packet received - skipping")
                        continue

                    parsed = await self.try_parse_data(data)

                    if parsed is None:
                        continue

                    if not await self.validate_data_structure(parsed):
                        print(f"[-] Invalid data structure: {parsed.keys()}")
                        continue

                    measurements = await self.extract_measurements(parsed)

                    if not measurements:
                        print("[-] Empty measurements - skipping")
                        continue

                    is_attack, attack_prob = await self.detect_fdia(measurements)

                    # --- Ground truth handling ---
                    ground_truth = parsed.get('metadata', {}).get('ground_truth')
                    if ground_truth is not None:
                        self.y_true.append(int(ground_truth))
                        self.y_pred.append(1 if is_attack else 0)

                    response = {
                        'data': {
                            'V': parsed.get('V', parsed.get('data', {}).get('V', [])),
                            'I': parsed.get('I', parsed.get('data', {}).get('I', []))
                        },
                        'timestamp': time.time(),
                        'detection': {
                            'is_attack': is_attack,
                            'probability': attack_prob
                        },
                        'metadata': parsed.get('metadata', {})
                    }

                    await websocket.send(json.dumps(response))
                    print(f"[+] Processed data and sent response. Attack={is_attack}, Score={attack_prob:.4f}")

                except websockets.exceptions.ConnectionClosed:
                    print("[-] WebSocket client disconnected")
                    break
                except Exception as e:
                    print(f"[-] Processing error: {str(e)}")
                    await asyncio.sleep(0.1)

        except Exception as e:
            print(f"[-] WebSocket handler error: {str(e)}")

    async def run_websocket(self):
        async with websockets.serve(
            self.handle_websocket,
            "0.0.0.0",
            self.ws_port,
            ping_interval=None
        ):
            print(f"[+] WebSocket server started on port {self.ws_port}")
            await asyncio.Future()

    def print_metrics(self):
        if len(self.y_true) == 0:
            print("[!] No ground_truth collected yet")
            return
        print("\n=== FDIA Detection Metrics ===")
        print(f"Accuracy : {accuracy_score(self.y_true, self.y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_true, self.y_pred):.4f}")
        print(f"Recall   : {recall_score(self.y_true, self.y_pred):.4f}")
        print(f"F1 Score : {f1_score(self.y_true, self.y_pred):.4f}")
        print(classification_report(self.y_true, self.y_pred))

    def run(self):
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.run_websocket())
            loop.close()
        except KeyboardInterrupt:
            print("[+] Server shutdown requested")
        except Exception as e:
            print(f"[-] Server error: {str(e)}")
        finally:
            print("[+] Closing UDP socket")
            self.udp_sock.close()
            print("[+] Server shutdown complete")
            self.print_metrics()


if __name__ == "__main__":
    bridge = UDPServer()
    bridge.run()
