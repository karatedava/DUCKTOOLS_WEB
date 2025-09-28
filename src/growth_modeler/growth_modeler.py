"""
DEFINITION OF GROWTH MODELER 
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.simulator import Simulator

class GrowthModeler(Simulator):

    def __init__(self, config_path:Path, data_dir:Path):

        super().__init__(config_path, data_dir)

    def run_simulation(self) -> Path:

        """
        Runs grid search simulation
        - returns path to report file for further statistical analysis
        """

        fname = f'GM_ct.{self.parameters['cultivation_time']}.WL.{self.parameters['limiting_biomass']}.r0.{self.parameters['initial_growth_rate']}'
        filepath = self.data_dir / fname

        header = 'HP\tHR\tIDENS\tHARVEST\n'
        report = header

        best_harvest = 0.0
        best_biomass_df = None
        best_harvest_df = None

        for hp in self.HPs:
            for hr in self.HRs:
                for w0 in self.W0s:

                    harvested, biomass_df, harvest_df = self.run_single(
                        HP=hp,
                        HR=hr,
                        W0=w0
                    )

                    line = f'{hp}\t{hr:.2}\t{w0:.2f}\t{harvested:.3f}\n'
                    report += line

                    if harvested > best_harvest:
                        best_harvest = harvested
                        best_biomass_df = biomass_df
                        best_harvest_df = harvest_df

        
        open(filepath,'w').write(report)
        best_biomass_df.to_csv(str(filepath) + '.biomass_df.csv',index=False)
        best_harvest_df.to_csv(str(filepath) + '.harvest_df.csv',index=False)

        return fname

    def run_single(self, HP, HR, W0):

        """
        run single simulation with single set of parameters
        """

        dt = self.parameters['integration_step']
        cultivation_time = self.parameters['cultivation_time']
        times = np.arange(0,cultivation_time + dt, dt)
        biomass_values = []
        harvest_values = []
        harvest_times = []
        storage = 0.0

        eff_gr = self._effective_gr_()
        next_harvest_time = HP
        Biomass = W0
        for t in times:
            if t >= next_harvest_time:
                harvested = HR * Biomass
                storage += harvested
                Biomass -= harvested
                next_harvest_time += HP
                harvest_values.append(harvested)
                harvest_times.append(t)
            
            dW = self._dW_(eff_gr,Biomass)
            Biomass += dW
            biomass_values.append(Biomass)
        
        # harvest all what is left 
        storage += Biomass

        biomass_df = pd.DataFrame({
            'biomass':biomass_values,
            'time':times
        })
        harvest_df = pd.DataFrame({
            'biomass':harvest_values,
            'time':harvest_times
        })

        return storage, biomass_df, harvest_df
    
    def _dW_(self, eff_gr, Biomass):

        dt = self.parameters['integration_step']
        WL = self.parameters['limiting_biomass']

        return eff_gr * Biomass * (1 - Biomass / WL) * dt