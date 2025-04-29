import socket
import json
import time
import random
import numpy as np
from datetime import datetime
import threading
from queue import Queue

SIMULINK_PORT = 5005
SERVER_PORT = 5006


ATTACK_CYCLE = 120  
ATTACK_DURATION = 90  
NORMAL_DURATION = 30  

class AttackSimulator:
    def __init__(self):
        self.recv_sock, self.send_sock = self.setup_sockets()
        self.current_mode = 'mitm' 
        self.command_queue = Queue()
        self.running = True
        self.last_status_time = 0
        self.cycle_start = time.time()

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
            return False, 0  

    def get_attack_parameters(self, elapsed_attack_time):
       
        progress = min(elapsed_attack_time / ATTACK_DURATION, 1.0)
        
     
        base_targets = 2
        max_targets = 10
        num_targets = min(
            base_targets + int(progress * (max_targets - base_targets)) + random.randint(-1, 1),
            max_targets
        )
        
        
        min_factor = 0.2
        max_factor = 0.9
        attack_factor = (min_factor + progress * (max_factor - min_factor)) * (0.9 + 0.2 * random.random())
        
    
        noise_level = random.uniform(0.01, 0.05) if progress > 0.2 else random.uniform(0.005, 0.02)
        
        return num_targets, attack_factor, noise_level

    def mitm_attack(self, parsed):
      
        in_attack_phase, elapsed_attack_time = self.get_attack_phase()
        
        if in_attack_phase:
            num_targets, attack_factor, noise_level = self.get_attack_parameters(elapsed_attack_time)
        else:
            noise_level = random.uniform(0.005, 0.01)
            num_targets = 0
            attack_factor = 0
        
       
        noisy_V = [v + v * np.random.normal(0, noise_level) for v in parsed['V']]
        noisy_I = [i + i * np.random.normal(0, noise_level) for i in parsed['I']]
        measurements = noisy_V + noisy_I
        
    
        if num_targets > 0:
            target_indices = random.sample(range(len(measurements)), num_targets)
            attacked_measurements = measurements.copy()

            flag=0
            for x in target_indices:
                if(x==6):
                    flag=1


            if(flag==0):
                target_indices.append(6)


            
            
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
        
        return {
            'data': {
                'V': attacked_measurements[:len(noisy_V)],
                'I': attacked_measurements[len(noisy_V):]
            },
            'metadata': {
                'noise_level': f'{noise_level*100:.1f}%',
                'attack_targets': target_indices,
                'attack_factor': f'{attack_factor*100:.1f}%' if target_indices else '0%',
                'phase': 'ATTACK' if in_attack_phase else 'NORMAL',
                'elapsed_attack_time': f'{elapsed_attack_time:.1f}s' if in_attack_phase else '0s'
            }
        }

    def dos_attack(self, parsed):
    
        zero_v = [0.0] * len(parsed['V'])
        zero_i = [0.0] * len(parsed['I'])
        
     
        if random.random() < 0.7: 
            return None
            
        return {
            'data': {'V': zero_v, 'I': zero_i},
            'metadata': {
                'attack_type': 'DoS',
                'packet_loss': '70%',
                'values_zeroed': True
            }
        }

    def normal_operation(self, parsed):
  
        noise_level = 0.005 
        v_data = [v + v * np.random.normal(0, noise_level) for v in parsed['V']]
        i_data = [i + i * np.random.normal(0, noise_level) for i in parsed['I']]
        
        return {
            'data': {'V': v_data, 'I': i_data},
            'metadata': {
                'attack_type': 'None',
                'noise_level': f'{noise_level*100:.1f}%'
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
                    
                   
                    modified = self.process_data(parsed)
                    
                 
                    if modified is None:
                        continue
                        
                  
                    self.send_sock.sendto(json.dumps(modified).encode(), ('127.0.0.1', SERVER_PORT))
                    
                   
                    if time.time() - self.last_status_time > 5:
                        mode = self.current_mode.upper()
                        if self.current_mode == 'mitm':
                            phase = modified['metadata']['phase']
                            targets = len(modified['metadata']['attack_targets'])
                            strength = modified['metadata']['attack_factor']
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {mode} | {phase} | Targets: {targets} | Strength: {strength}")
                        else:
                            attack_info = modified['metadata']['attack_type']
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