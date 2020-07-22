import spotpy
import numpy as np
import pandas as pd
from run_project import *


class SpotpySetup(object):
    """

    """
    def __init__(self, proj_dir, start_date, end_date, obs, date_format='%Y-%m-%d'):
        """

        Args:
            proj_dir:
            start_date:
            end_date:
            obs:
            date_format:
        """
        self.proj_dir = proj_dir
        self.start_date = start_date
        self.end_date = end_date
        self.obs = self.process_observations(obs)
        self.params = [spotpy.parameter.Uniform('s', 0, 500, 50, 200),
                       spotpy.parameter.Uniform('bk', 0.001, 0.1, 0.01, 0.04),
                       spotpy.parameter.Uniform('ds', 0.001, 0.1, 0.01, 0.01),
                       spotpy.parameter.Uniform('ba', 1.0, 1.0, 1.0, 1.0)]

    def evaluation(self):
        """

        Returns:

        """
        return self.obs

    def parameters(self):
        """

        Returns:

        """
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        """

        Args:
            vector:

        Returns:

        """
        run_project(self.proj_dir, numcpu=1, gwcoeff=vector)
        fn_wepp = self.proj_dir + '/wepp/output/chnwb.txt'
        df_wepp = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=25, header=None)
        colnames_units = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=21, header=0, nrows=1)
        df_wepp.columns = colnames_units.columns
        df_wepp['date'] = pd.to_datetime(df_wepp['Y'] * 1000 + df_wepp['J'], format='%Y%j')
        df_wepp['Qvol'] = (df_wepp['Q'] / 1000.0) * df_wepp['Area']
        df_wepp['Qday'] = (df_wepp['Qvol'] / (3600 * 24)) / 0.0283168  # cfs

        df_wepp = df_wepp.loc[(df_wepp['date'] >= self.start_date) & (df_wepp['date'] <= self.end_date)]
        df_wepp = df_wepp.loc[df_wepp['OFE'] == df_wepp['OFE'].max()]
        return df_wepp['Qday'].to_numpy()

    def process_observations(self, obs, date_col='DATE', q_col='MEAN_Q'):
        """

        Args:
            obs:
            date_col:
            q_col:

        Returns:

        """
        obs = obs.loc[(obs[date_col] >= self.start_date) & (obs[date_col] <= self.end_date)]
        return obs[q_col].to_numpy()
