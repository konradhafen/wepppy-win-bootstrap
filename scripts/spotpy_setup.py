import spotpy
import pandas as pd
import numpy as np
import logging
import os
from run_project import *


def water_year_yield(df, date_col='DATE', q_col='MEAN_Q', conv=1.0):
    years = pd.DatetimeIndex(df[date_col]).year.unique()
    vol = []
    for i in range(len(years) - 1):
        start_date = str(years[i]) + '-10-01'
        end_date = str(years[i+1]) + '-09-30'
        tmp_dat = df.loc[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
        vol.append(tmp_dat[q_col].sum() * conv)  # sum and convert units
    df_wy = pd.DataFrame({'year': years[:-1], 'yield_m3': vol})
    return df_wy


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
        self.logger = logging.getLogger('spotpy_setup')
        self.logger.setLevel(logging.INFO)
        self.proj_dir = proj_dir
        self.logger.info('project directory ' + str(self.proj_dir))
        self.start_date = start_date
        self.end_date = end_date
        self.obs = self.process_observations(obs)
        self.logger.info('obs shape ' + str(self.obs.shape))
        self.params = [spotpy.parameter.Uniform('kb', 0.001, 0.1, 0.01, 0.04),
                       spotpy.parameter.Uniform('ks', 0.0, 0.1, 0.01, 0.01),
                       spotpy.parameter.Uniform('ksat_fact', 0.1, 10.0, 0.1, 1.0),
                       spotpy.parameter.Uniform('kr', 50.0, 1000.0, 20.0, 144.0)]

    def evaluation(self):
        """

        Returns:

        """
        return self.obs

    def objectivefunction(self, simulation, evaluation):
        """

        Args:
            simulation:
            evaluation:

        Returns:

        """
        self.logger.info('in objective function ' + str(type(simulation)) + str(type(evaluation)))
        objectivefunction = spotpy.objectivefunctions.nashsutcliffe(evaluation, simulation)
        return objectivefunction

    def parameters(self):
        """

        Returns:

        """
        return spotpy.parameter.generate(self.params)

    def process_observations(self, obs, date_col='DATE', q_col='MEAN_Q'):
        """

        Args:
            obs:
            date_col:
            q_col:

        Returns:

        """
        obs = obs.loc[(obs[date_col] >= self.start_date) & (obs[date_col] <= self.end_date)]
        return obs[q_col].to_numpy(np.float32)

    def simulation(self, vector):
        """

        Args:
            vector:

        Returns:

        """
        self.logger.info('running simulation ' + str(vector))
        gwcoeffs = [200.0, vector[0], vector[1], 1.0001]  # initial storage, baseflow recession, deep seepage, minimum area
        gwcoeffs = [200.0, 0.04, 0.0, 1.0001]  # hard-coded for testing, remove this for sensitivity and calibration
        result = run_project(self.proj_dir, numcpu=8, gwcoeff=gwcoeffs)
        self.logger.info('simulation complete')
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


class SpotpySetupAnnual():
    """

    """
    def __init__(self, proj_dir, start_year, end_year, obs, date_format='%Y-%m-%d'):
        """

        Args:
            proj_dir:
            start_date:
            end_date:
            obs:
            date_format:
        """
        self.logger = logging.getLogger('spotpy_setup')
        self.logger.setLevel(logging.INFO)
        self.proj_dir = proj_dir
        self.logger.info('project directory ' + str(self.proj_dir))
        self.start_year = start_year
        self.end_year = end_year
        self.obs = self.process_observations(obs)
        self.logger.info('obs shape ' + str(self.obs.shape))
        self.params = [spotpy.parameter.Uniform('kc', 1.0, 1.0, 0.01, 1.0)]  # crop coefficient
        self.database = open(os.path.join(self.proj_dir, 'export/calibration_results_annual.csv'), 'w')

    def evaluation(self):
        return self.obs

    def objectivefunction(self, simulation, evaluation):
        # self.logger.info('sim ' + str(simulation))
        # self.logger.info('eval ' + str(evaluation))
        objectivefunction = spotpy.objectivefunctions.pbias(evaluation, simulation)
        return objectivefunction

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def process_observations(self, obs, col_name='yield_m3'):
        # self.logger.info(str(obs.head()))
        evaluation = water_year_yield(obs, conv=3600 * 24 * 0.0283168)  # convert from cfs to cubic meters
        evaluation = evaluation.loc[(evaluation['year'] >= self.start_year) & (evaluation['year'] <= self.end_year)]
        # self.logger.info(str(evaluation))
        return evaluation[col_name].to_numpy()

    def save(self, objectivefunctions, parameter, simulations):
        line = str(objectivefunctions)+','+str(parameter).strip('[]')+','+str(simulations.tolist()).strip('[]')+'\n'
        self.logger.info('line' + line)
        self.database.write(line)

    def simulation(self, vector):
        self.logger.info('running simulation ' + str(vector))
        pmet_coeffs = [vector[0], 0.8]
        result = run_project(self.proj_dir, numcpu=8, pmet=pmet_coeffs)
        self.logger.info('simulation complete ' + str(result))
        fn_wepp = self.proj_dir + '/wepp/output/chnwb.txt'
        df_wepp = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=25, header=None)
        colnames_units = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=21, header=0, nrows=1)
        df_wepp.columns = colnames_units.columns
        df_wepp['date'] = pd.to_datetime(df_wepp['Y'] * 1000 + df_wepp['J'], format='%Y%j')
        df_wepp = df_wepp.loc[df_wepp['OFE'] == df_wepp['OFE'].max()]
        df_wepp['Qvol'] = (df_wepp['Q'] / 1000.0) * df_wepp['Area']
        # df_wepp['Qday'] = (df_wepp['Qvol'] / (3600 * 24)) / 0.0283168  # cfs
        df_mod = water_year_yield(df_wepp, 'date', 'Qvol')
        df_mod = df_mod.loc[(df_mod['year'] >= self.start_year) & (df_mod['year'] <= self.end_year)]
        # self.logger.info(str(df_mod))

        return df_mod['yield_m3'].to_numpy()