"""
power.py

GPU power monitoring using nvidia-smi.
"""

import subprocess
import threading
import time
from typing import List 

class GPUPowerMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval 
        self._running = False 
        self.samples: List[float] = []
        self.timestamps: List[float] = []
        self._thread = None
        
    def _poll(self):
        while self._running:
            try:
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw", # Assumes single-GPU environment
                        "--format=csv,noheader, nounits",
                    ],
                    encoding = "utf-8",
                )
                power = float(result.strip().split("\n")[0])
                self.samples.append(power)
                self.timestamps.append(time.time())
            except Exception:
                pass 
            
            time.sleep(self.interval)
            
    def start(self):
        self._running = True
        self._thread = threading.Thread(target = self._poll, daemon = True)
        self._thread.start()
        
    def stop(self):
        self._running = False 
        if self._thread is not None:
            self._thread.join()
            
    def energy_wh(self):
        """
        Integrate power over time (Wh)
        """
        if len(self.samples) < 2:
            return 0.0 
        
        energy = 0.0 
        for i in range(1, len(self.samples)):
            dt = self.timestamps[i] - self.timestamps[i-1]
            energy += self.samples[i] * dt 
            
        return energy / 3600.0  # joules -> Wh