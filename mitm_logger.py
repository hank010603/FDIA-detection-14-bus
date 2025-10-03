# mitm_logger.py
import socket
import json
import time
import random
import numpy as np
from datetime import datetime
import threading
from queue import Queue
import csv
import os

SIMULINK_PORT = 5005
SERVER_PORT = 5006
NUM_BUSES = 42

ATTACK_CYCLE = 120
ATTACK_DURATION = 90
NORMAL_DURATION = 30

CSV_PATH = "mitm_output_stream.csv"   # 輸出的 CSV 檔名

class AttackSimulator:
    def __init__(self, csv_path=CSV_PATH):
        self.recv_sock, self.send_sock = self.setup_sockets()
        self.current_mode = 'mitm'
        self.command_queue = Queue()
        self.running = True
        self.last_status_time = 0
        self.cycle_start = time.time()

        # CSV logger init
        self.csv_path = csv_path
        self.csv_lock = threading.Lock()
        self.ensure_csv_header()

    def ensure_csv_header(self):
        """Ensure CSV exists and has header"""
        header = [f'V_{i}' for i in range(NUM_BUSES)] + [f'I_{i}' for i in range(NUM_BUSES)]
        header += ['ground_truth', 'timestamp', 'metadata_json']
        # create file and write header if not exists
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        else:
            # optionally check header length / name; skip for speed
            pass

    def write_row_to_csv(self, v_list, i_list, ground_truth, metadata):
        """Append one row to CSV. v_list and i_list should be length NUM_BUSES each."""
        if len(v_list) != NUM_BUSES or len(i_list) != NUM_BUSES:
            # skip invalid rows
            return
        row = list(v_list) + list(i_list) + [int(ground_truth), time.time(), json.dumps(metadata, ensure_ascii=False)]
        try:
            with self.csv_lock:
                with open(self.csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
        except Exception as e:
            print("[!] CSV write error:", e)

    def setup_sockets(self):
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(('127.0.0.1', SIMULINK_PORT))
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return recv_sock, send_sock

    def get_attack_phase(self):
        cycle_elapsed = time.time() - self.cycle_start
        cycle_position = cycle_elapsed % ATTACK_CYCLE
        if cycle_position < ATTACK_DURATION:
            return True, cycle_position
        else:
            return False, 0.0

    def get_attack_parameters(self, elapsed_attack_time):
        progress = min(elapsed_attack_time / ATTACK_DURATION, 1.0)
        base_targets = 2
        max_targets = 10
        num_targets = min(
            base_targets + int(progress * (max_targets - base_targets)) + random.randint(-1, 1),
            max_targets
        )
        min_factor = 0.05
        max_factor = 0.35
        attack_factor = (min_factor + progress * (max_factor - min_factor)) * (0.8 + 0.4 * random.random())
        noise_level = random.uniform(0.01, 0.03) if progress > 0.2 else random.uniform(0.005, 0.02)
        return num_targets, attack_factor, noise_level

    def generate_measurements(self):
        V = np.random.uniform(0.95, 1.05, NUM_BUSES).tolist()
        I = np.random.uniform(-0.05, 0.05, NUM_BUSES).tolist()
        return {'V': V, 'I': I}

    def mitm_attack(self, parsed):
        in_attack_phase, elapsed_attack_time = self.get_attack_phase()
        p_attack_chance = 0.8
        will_attack = in_attack_phase and (random.random() < p_attack_chance)

        if will_attack:
            num_targets, attack_factor, noise_level = self.get_attack_parameters(elapsed_attack_time)
        else:
            num_targets, attack_factor = 0, 0.0
            noise_level = random.uniform(0.01, 0.03) if not in_attack_phase else random.uniform(0.005, 0.02)

        noisy_V = [v + v * np.random.normal(0, noise_level) for v in parsed['V']]
        noisy_I = [i + i * np.random.normal(0, noise_level) for i in parsed['I']]
        measurements = noisy_V + noisy_I

        if num_targets > 0 and num_targets <= len(measurements):
            target_indices = random.sample(range(len(measurements)), num_targets)
            if 6 not in target_indices and random.random() < 0.7:
                if len(target_indices) < len(measurements):
                    target_indices.append(6)

            attacked_measurements = measurements.copy()
            for idx in target_indices:
                attack_type = random.choice(['constant', 'progressive', 'oscillating'])
                if attack_type == 'constant':
                    modification = random.choice([-1, 1]) * abs(measurements[idx]) * attack_factor
                elif attack_type == 'progressive':
                    mod_factor = attack_factor * (0.5 + 0.5 * (elapsed_attack_time / ATTACK_DURATION))
                    modification = random.choice([-1, 1]) * abs(measurements[idx]) * mod_factor
                else:
                    oscillation = np.sin(elapsed_attack_time * 0.1) * 0.5 + 0.5
                    modification = random.choice([-1, 1]) * abs(measurements[idx]) * attack_factor * oscillation
                attacked_measurements[idx] += modification
        else:
            target_indices = []
            attacked_measurements = measurements

        ground_truth = 1 if (will_attack and len(target_indices) > 0) else 0

        return {
            'data': {
                'V': attacked_measurements[:NUM_BUSES],
                'I': attacked_measurements[NUM_BUSES:]
            },
            'metadata': {
                'ground_truth': ground_truth,
                'noise_level': f'{noise_level*100:.2f}%',
                'attack_targets': target_indices,
                'attack_factor': f'{attack_factor*100:.2f}%' if target_indices else '0%',
                'phase': 'ATTACK' if will_attack else ('NORMAL' if not in_attack_phase else 'ATTACK_SKIPPED'),
                'elapsed_attack_time': f'{elapsed_attack_time:.1f}s' if in_attack_phase else '0s'
            }
        }

    def dos_attack(self, parsed):
        zero_v = [0.0] * NUM_BUSES
        zero_i = [0.0] * NUM_BUSES
        if random.random() < 0.5:
            return None
        return {
            'data': {'V': zero_v, 'I': zero_i},
            'metadata': {
                'ground_truth': 1,
                'attack_type': 'DoS',
                'packet_loss': 'approx_50%',
                'values_zeroed': True
            }
        }

    def normal_operation(self, parsed):
        noise = random.uniform(0.01, 0.03)
        V = [v + v * np.random.normal(0, noise) for v in parsed['V']]
        I = [i + i * np.random.normal(0, noise) for i in parsed['I']]
        return {
            'data': {'V': V, 'I': I},
            'metadata': {
                'ground_truth': 0,
                'attack_type': 'None',
                'noise_level': f'{noise*100:.2f}%'
            }
        }

    def process_data(self, parsed):
        if self.current_mode == 'mitm':
            return self.mitm_attack(parsed)
        elif self.current_mode == 'dos':
            return self.dos_attack(parsed)
        else:
            return self.normal_operation(parsed)

    def handle_keyboard_input(self):
        print("\nCommand controls:")
        print("  m - MITM attack (auto cycle attack/normal)")
        print("  d - DoS attack (zero values, packet loss)")
        print("  n - Normal operation")
        print("  q - Exit program")
        while self.running:
            try:
                cmd = input("\nEnter command (m/d/n/q): ").lower()
                if cmd in ['m', 'd', 'n', 'q']:
                    self.command_queue.put(cmd)
                    if cmd == 'q':
                        break
            except:
                break

    def run(self):
        print(f"[+] Starting simulator - Default mode: MITM (auto cycle)")
        print(f"[+] Listening on port {SIMULINK_PORT}, forwarding to port {SERVER_PORT}")
        print(f"[+] MITM cycle: {ATTACK_DURATION}s attack, {NORMAL_DURATION}s normal")

        input_thread = threading.Thread(target=self.handle_keyboard_input, daemon=True)
        input_thread.start()

        while self.running:
            while not self.command_queue.empty():
                cmd = self.command_queue.get()
                if cmd == 'm':
                    self.current_mode = 'mitm'
                    self.cycle_start = time.time()
                    print("\n[!] ACTIVATED MITM ATTACK (auto cycle)")
                elif cmd == 'd':
                    self.current_mode = 'dos'
                    print("\n[!] ACTIVATED DoS ATTACK")
                elif cmd == 'n':
                    self.current_mode = 'normal'
                    print("\n[!] NORMAL OPERATION RESTORED")
                elif cmd == 'q':
                    self.running = False
                    print("\n[!] SHUTTING DOWN...")
                    break

            try:
                self.recv_sock.settimeout(0.1)
                data, addr = self.recv_sock.recvfrom(4096)
                try:
                    decoded = data.decode().rstrip('\x00').strip()
                    parsed = json.loads(decoded)
                    if ('V' not in parsed or 'I' not in parsed or
                            len(parsed.get('V', [])) != NUM_BUSES or
                            len(parsed.get('I', [])) != NUM_BUSES):
                        parsed = self.generate_measurements()

                    modified = self.process_data(parsed)
                    if modified is None:
                        continue

                    # 轉發到 server
                    self.send_sock.sendto(json.dumps(modified).encode(), ('127.0.0.1', SERVER_PORT))

                    # --- 寫 CSV（在成功轉發後） ---
                    try:
                        v_out = modified['data']['V']
                        i_out = modified['data']['I']
                        ground = modified['metadata'].get('ground_truth', 0)
                        meta = {k: v for k, v in modified['metadata'].items()}
                        # 寫入 CSV
                        self.write_row_to_csv(v_out, i_out, ground, meta)
                    except Exception as e:
                        print("[!] CSV logging error:", e)

                    # 每隔一段時間印出狀態
                    if time.time() - self.last_status_time > 5:
                        mode = self.current_mode.upper()
                        if self.current_mode == 'mitm':
                            phase = modified['metadata']['phase']
                            targets = len(modified['metadata']['attack_targets'])
                            strength = modified['metadata']['attack_factor']
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {mode} | {phase} | Targets: {targets} | Strength: {strength}")
                        else:
                            attack_info = modified['metadata'].get('attack_type', '')
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Mode: {mode} | {attack_info}")
                        self.last_status_time = time.time()
                except json.JSONDecodeError:
                    self.send_sock.sendto(data, ('127.0.0.1', SERVER_PORT))
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[!] Error: {str(e)}")
                time.sleep(0.1)

        self.recv_sock.close()
        self.send_sock.close()
        print("[!] Simulation stopped")


if __name__ == "__main__":
    simulator = AttackSimulator()
    simulator.run()
