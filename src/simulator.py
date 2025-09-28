from abc import abstractmethod
from pathlib import Path
import numpy as np
import json

class Simulator():

    def __init__(self, config_path:Path, data_dir:Path):
        
        self.parameters = json.load(config_path.open())
        self.data_dir = data_dir

        self.HPs = np.arange(1,10,1)
        self.HRs = np.arange(0.1,0.8,0.1)
        self.W0s = np.arange(100,1000,100)
        
    def _effective_gr_(self) -> float:

        r0 = self.parameters['initial_growth_rate'] 
        alpha = self.parameters['alpha'] 
        beta = self.parameters['beta'] 
        gamma = self.parameters['gamma'] 
        delta = self.parameters['delta'] 

        eff_gr = r0 * alpha * beta * gamma * delta

        return eff_gr

    @abstractmethod
    def run_single(self, HP, HR, W0) -> float:
        pass

    @abstractmethod
    def run_simulation(self) -> Path:
        pass