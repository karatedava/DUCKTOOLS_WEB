"""
DEFINITION OF THE YIELD PREDICTOR
- designed for long-term cultivation design only
- minimal recommended timespan --> month 
"""
import torch
import numpy as np

from pathlib import Path
import joblib

from src.simulator import Simulator

DEPENDENCIES_PATH = Path('./src/yield_predictor/dependencies')
WET_DRY_COEF = 16.66667

class YieldPredictor(Simulator):

    def __init__(self, config_path:Path, data_dir:Path):

        super().__init__(config_path, data_dir)
    
        self.model = torch.jit.load(DEPENDENCIES_PATH / 'harvest_model.pt').to('cpu')
        self.model.eval()
        self.scaler = joblib.load(DEPENDENCIES_PATH / 'HP_scaler.save')
    
    def run_simulation(self) -> Path:

        fname = f'YP_ct.year.WL.{self.parameters['limiting_biomass']}.r0.{self.parameters['initial_growth_rate']}'
        filepath = self.data_dir / fname

        header = 'HP\tHR\tIDENS\tHARVEST\n'
        report = header

        for hp in self.HPs:
            for hr in self.HRs:
                for w0 in self.W0s:

                    harvested = self.run_single(
                        HP=hp,
                        HR=hr,
                        W0=w0
                    )

                    line = f'{hp}\t{hr:.2}\t{w0:.2f}\t{harvested:.3f}\n'
                    report += line
        
        open(filepath,'w').write(report)

        return fname

    @torch.no_grad
    def run_single(self, HP, HR, W0) -> float:

        eff_gr = self._effective_gr_()
        exp_gr = np.exp(eff_gr)
        # convert WET to DRY
        W0 = W0 / WET_DRY_COEF

        W0_expR = W0 * np.exp(eff_gr)
        R_F = HR / HP

        X = np.array([W0_expR,R_F,HP,eff_gr,exp_gr,HR])
        X = X.reshape((1,6))
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        predicted_yield = self.model(X).item()

        # return WET again
        return predicted_yield * WET_DRY_COEF