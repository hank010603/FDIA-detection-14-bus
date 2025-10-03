# udp_send.py  (改良版)
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

SIMULINK_PORT = 5006  # UDPServer listening port (from AttackSimulator forwarder)
WS_PORT = 8765
NUM_BUSES = 42  # 與你的模擬器一致

class UDPServer:
    def __init__(self, udp_port: int = SIMULINK_PORT, ws_port: int = WS_PORT):
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
            # 支援多種儲存格式
            if isinstance(model_data, dict):
                self.fdia_model = model_data.get("model", None)
                self.preprocessor = model_data.get("preprocessor", None)
                print(f"[+] Loaded dict bundle from {model_path}. Model: {type(self.fdia_model)}, Preprocessor: {type(self.preprocessor)}")
            else:
                # 可能直接存了 estimator (或 pipeline)
                self.fdia_model = model_data
                self.preprocessor = None
                print(f"[+] Loaded estimator from {model_path}: {type(self.fdia_model)}")
        except Exception as e:
            print(f"[!] Error loading model '{model_path}': {e}")
            self.fdia_model = None
            self.preprocessor = None

        print(f"[+] UDP server initialized on port {udp_port}")
        print(f"[+] WebSocket server will run on port {ws_port}")

        if sys.platform == 'win32':
            self.udp_sock.setblocking(False)

        # --- Metrics storage ---
        self.y_true = []
        self.y_pred = []

        # Debug: run a quick model sanity check on random data
        self.debug_model_on_dummy()

    def debug_model_on_dummy(self, n=20):
        """用隨機測量測試 model 行為，觀察 predict / predict_proba 分布"""
        if not self.fdia_model:
            print("[DEBUG] No model loaded for debug -> skipping debug_model_on_dummy()")
            return
        try:
            X = np.random.uniform(0.95, 1.05, size=(n, NUM_BUSES))  # V
            X2 = np.random.uniform(-0.05, 0.05, size=(n, NUM_BUSES))  # I
            X_full = np.hstack([X, X2])  # shape (n, 2*NUM_BUSES)

            print(f"[DEBUG] Dummy input shape: {X_full.shape}, expected cols: {NUM_BUSES*2}")

            Xp = X_full
            if self.preprocessor is not None:
                try:
                    Xp = self.preprocessor.transform(X_full)
                except Exception as e:
                    print("[DEBUG] Preprocessor transform error during debug:", e)
                    Xp = X_full

            preds = None
            try:
                preds = self.fdia_model.predict(Xp)
                unique, counts = np.unique(preds, return_counts=True)
                print("[DEBUG] preds unique & counts:", dict(zip(unique.tolist(), counts.tolist())))
            except Exception as e:
                print("[DEBUG] Model predict error during debug:", e)
                return

            if hasattr(self.fdia_model, "predict_proba"):
                try:
                    probs = self.fdia_model.predict_proba(Xp)[:, 1]
                    print("[DEBUG] probs min/mean/max:", float(probs.min()), float(probs.mean()), float(probs.max()))
                except Exception as e:
                    print("[DEBUG] predict_proba error during debug:", e)
            elif hasattr(self.fdia_model, "decision_function"):
                try:
                    scores = self.fdia_model.decision_function(Xp)
                    print("[DEBUG] decision_function min/mean/max:", float(np.min(scores)), float(np.mean(scores)), float(np.max(scores)))
                except Exception as e:
                    print("[DEBUG] decision_function error during debug:", e)
            else:
                print("[DEBUG] Model has no predict_proba or decision_function; using raw preds as proxy.")
        except Exception as e:
            print("[DEBUG] debug_model_on_dummy general error:", e)

    async def detect_fdia(self, measurements):
        """FDIA detection using joblib model + optional preprocessor (robust)"""
        if not self.fdia_model or not measurements:
            return False, 0.0

        try:
            features = np.array(measurements).reshape(1, -1)
            expected_cols = NUM_BUSES * 2
            # 檢查長度
            if features.shape[1] != expected_cols:
                print(f"[!] Feature length mismatch: got {features.shape[1]}, expected {expected_cols}")
                # 嘗試修補：如果只有 V 或只有 I，嘗試補零
                if features.shape[1] == NUM_BUSES:
                    print("[!] Detected single-side features (only V or only I). Padding zeros for missing half.")
                    pad = np.zeros((1, expected_cols - features.shape[1]))
                    features = np.hstack([features, pad])
                else:
                    # 若長度不對，直接返回 False
                    return False, 0.0

            # preprocessor
            if self.preprocessor is not None:
                try:
                    features = self.preprocessor.transform(features)
                except Exception as e:
                    print(f"[!] Preprocessor transform error: {e}")

            # predict
            pred = None
            try:
                pred = self.fdia_model.predict(features)
            except Exception as e:
                print(f"[!] Model predict error: {e}")
                return False, 0.0

            # safe convert prediction to bool
            try:
                is_attack = bool(int(pred[0]))
            except Exception:
                try:
                    is_attack = bool(pred[0])
                except Exception:
                    is_attack = False

            # probability attempt
            prob = 0.0
            if hasattr(self.fdia_model, "predict_proba"):
                try:
                    prob = float(self.fdia_model.predict_proba(features)[0][1])
                except Exception as e:
                    print(f"[!] predict_proba error: {e}")
                    prob = float(is_attack)
            else:
                # if model doesn't have predict_proba, try decision_function
                if hasattr(self.fdia_model, "decision_function"):
                    try:
                        score = float(self.fdia_model.decision_function(features)[0])
                        # sigmoid to map to 0..1
                        prob = 1.0 / (1.0 + np.exp(-score))
                    except Exception as e:
                        print(f"[!] decision_function error: {e}")
                        prob = float(is_attack)
                else:
                    prob = float(is_attack)

            return is_attack, prob

        except Exception as e:
            print(f"FDIA detection error: {str(e)}")
            return False, 0.0

    async def receive_udp(self):
        loop = asyncio.get_event_loop()
        if sys.platform == 'win32':
            # non-blocking recv on windows
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
                    # receive_udp returns either (data, addr) or bytes depending on platform impl
                    if isinstance(data, tuple) and len(data) == 2:
                        # occasionally on some platforms loop.sock_recvfrom returns differently
                        data, addr = data

                    print(f"[+] Received {len(data)} bytes from {addr}")

                    if not data:
                        print("[-] Empty UDP packet received - skipping")
                        continue

                    parsed = await self.try_parse_data(data)

                    if parsed is None:
                        continue

                    if not await self.validate_data_structure(parsed):
                        print(f"[-] Invalid data structure: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
                        continue

                    measurements = await self.extract_measurements(parsed)

                    if not measurements:
                        print("[-] Empty measurements - skipping")
                        continue

                    is_attack, attack_prob = await self.detect_fdia(measurements)

                    # --- Ground truth handling ---
                    ground_truth = None
                    if isinstance(parsed, dict):
                        ground_truth = parsed.get('metadata', {}).get('ground_truth')
                    if ground_truth is not None:
                        try:
                            self.y_true.append(int(ground_truth))
                            self.y_pred.append(1 if is_attack else 0)
                        except Exception:
                            pass

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
        print(classification_report(self.y_true, self.y_pred, zero_division=0))

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
