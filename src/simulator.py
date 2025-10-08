from abc import abstractmethod
from pathlib import Path
import numpy as np
import json

class Simulator():

    def __init__(self, config_path:Path, data_dir:Path, HPs=np.arange(1,10,1), HRs=np.arange(0.1,0.8,0.1), W0s= np.arange(100,1000,100)):
        
        self.parameters = json.load(config_path.open())
        self.data_dir = data_dir

        self.HPs = HPs
        self.HRs = HRs
        self.W0s = W0s
        
    def _effective_gr_(self) -> float:

        r0 = self.parameters['initial_growth_rate'] 
        growth_const = self.parameters['growth_const']

        eff_gr = r0 * growth_const

        return eff_gr

    @abstractmethod
    def run_single(self, HP, HR, W0) -> float:
        pass

    @abstractmethod
    def run_simulation(self) -> Path:
        pass