import socket
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime

class FDIA_Dataset_Generator:
    def __init__(self, simulink_port=5005, num_buses=14):
        self.simulink_port = simulink_port
        self.num_buses = num_buses
        self.dataset = []
        self.sock = None
        
    def setup_socket(self):
      
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', self.simulink_port))
        print(f"[+] Listening for Simulink data on port {self.simulink_port}")

    def add_noise(self, values):
    
        noise_level = random.uniform(0.01, 0.02)
        return [v + v * np.random.normal(0, noise_level) for v in values]

    def inject_fdia(self, measurements):
    
        num_targets = random.randint(2, 5)
        target_indices = random.sample(range(len(measurements)), num_targets)
        
        attacked = measurements.copy()
        for idx in target_indices:
            attack_factor = random.uniform(0.15, 0.20)
            modification = random.choice([-1, 1]) * abs(measurements[idx]) * attack_factor
            attacked[idx] += modification
        return attacked, target_indices

    def process_packet(self, data, attack=False):
      
        try:
            
            decoded = data.decode().rstrip('\x00').strip()
            parsed = json.loads(decoded)
            
     
            V = parsed['V']
            I = parsed['I']
            
            if attack:
               
                noisy_V = self.add_noise(V)
                noisy_I = self.add_noise(I)
                measurements = noisy_V + noisy_I
                attacked_measurements, targets = self.inject_fdia(measurements)
              
                V_attacked = attacked_measurements[:len(noisy_V)]
                I_attacked = attacked_measurements[len(noisy_V):]
                
                return {
                    'V': V_attacked,
                    'I': I_attacked,
                    'label': 1,  
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'noise_level': '1-2%',
                        'attack_targets': targets,
                        'original_V': V,
                        'original_I': I
                    }
                }
            else:
             
                noise_level = random.uniform(0.005, 0.01)
                V_noisy = [v + v * np.random.normal(0, noise_level) for v in V]
                I_noisy = [i + i * np.random.normal(0, noise_level) for i in I]
                
                return {
                    'V': V_noisy,
                    'I': I_noisy,
                    'label': 0, 
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'noise_level': f'{noise_level*100:.1f}%',
                        'attack_targets': []
                    }
                }
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[-] Error processing packet: {e}")
            return None

    def collect_samples(self, num_samples=1000, attack_ratio=0.5):
      
        if not self.sock:
            self.setup_socket()
            
        print(f"[+] Collecting {num_samples} samples ({(attack_ratio*100):.0f}% attacks)")
        
        samples_collected = 0
        while samples_collected < num_samples:
            try:
               
                data, _ = self.sock.recvfrom(4096)
                
            
                attack = random.random() < attack_ratio
                
               
                sample = self.process_packet(data, attack=attack)
                if sample:
                    self.dataset.append(sample)
                    samples_collected += 1
                    
                
                    if samples_collected % 100 == 0:
                        print(f"Collected {samples_collected}/{num_samples} samples")
                        
            except KeyboardInterrupt:
                print("\n[!] Collection stopped by user")
                break
            except Exception as e:
                print(f"[!] Error: {e}")
                time.sleep(0.1)
                
        print("[+] Sample collection complete")
        return self.dataset

    def save_dataset(self, filename='fdia_dataset.json'):
      
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        print(f"[+] Dataset saved to {filename}")

    def convert_to_csv(self, json_file, csv_file):
      
        with open(json_file) as f:
            data = json.load(f)
        

        rows = []
        for sample in data:
            row = {
                **{f'V_{i}': v for i, v in enumerate(sample['V'])},
                **{f'I_{i}': c for i, c in enumerate(sample['I'])},
                'label': sample['label']
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        print(f"[+] CSV dataset saved to {csv_file}")
        return df

if __name__ == "__main__":
  
    generator = FDIA_Dataset_Generator(simulink_port=5005)
    
   
    dataset = generator.collect_samples(num_samples=500000, attack_ratio=0.5)
    
    generator.save_dataset('fdia_dataset_raw.json')
    generator.convert_to_csv('fdia_dataset_raw.json', 'fdia_dataset.csv')
    
 
    normal = sum(1 for sample in dataset if sample['label'] == 0)
    attacked = len(dataset) - normal
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Normal measurements: {normal} ({normal/len(dataset)*100:.1f}%)")
    print(f"Attacked measurements: {attacked} ({attacked/len(dataset)*100:.1f}%)")