import spotpy
import pandas as pd
import numpy as np
import logging
import os
from run_project import *
from sol_prep import *
from snow_prep import *


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
        obs = obs.loc[(obs['DATE'] >= self.start_date) & (obs['DATE'] <= self.end_date)]
        self.eval_water_year = water_year_yield(obs, conv=3600 * 24 * 0.0283168)  # convert from cfs to cubic meters
        # self.eval_water_year = self.eval_water_year.loc[(self.eval_water_year['year'] >= self.start_year) & (self.eval_water_year['year'] <= self.end_year)]
        self.eval_water_year = self.eval_water_year['yield_m3'].to_numpy()
        self.logger.info('obs shape ' + str(self.obs.shape))
        self.params = [spotpy.parameter.Uniform('kc', 0.6, 0.8, 0.01, 0.95),  # crop coefficient
                       spotpy.parameter.Uniform('kr', 0.0001, 0.5, 0.001, 0.05), # vertical conductivity of restrictive layer
                       spotpy.parameter.Uniform('ks', 0.001, 0.1, 0.01, 0.01),  # deep seepage coefficient
                       spotpy.parameter.Uniform('kb', 0.001, 0.1, 0.01, 0.01),  # baseflow coefficient
                       # spotpy.parameter.Uniform('fc', 0.0, 0.8, 0.01, 0.4),  # field capactiy
                       # spotpy.parameter.Uniform('pr', 0.0, 80.0, 0.11, 50.0),  # percent rock
                       ]

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
        # discard results from first year
        self.logger.info('\nIn objective function, eval and sim shapes ' + str(evaluation.shape) + str(simulation.shape))
        evaluation = evaluation[365:]
        simulation = simulation[365:]
        self.logger.info('After subset: ' + str(evaluation.shape) + str(simulation.shape))
        pb = spotpy.objectivefunctions.pbias(evaluation, simulation)
        objectivefunction = spotpy.objectivefunctions.nashsutcliffe(evaluation, simulation)
        obj_ln = spotpy.objectivefunctions.nashsutcliffe(np.log(evaluation + 0.0001), np.log(simulation + 0.0001))
        return [pb, objectivefunction, obj_ln]
        # return objectivefunction

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
        pmet_coeffs = [vector[0], 0.8]
        gw_coeffs = [200.0, vector[3], vector[2], 1.0001]
        snow_coeffs = [-2.0, 100.0, 250.0]
        soil_prep(self.proj_dir, kr=vector[1], field_cap=None, pct_rock=None)
        result = run_project(self.proj_dir, numcpu=8, gwcoeff=gw_coeffs, pmet=pmet_coeffs, snow=snow_coeffs)
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
        self.sim_water_year = water_year_yield(df_wepp, 'date', 'Qvol')
        self.sim_water_year = self.sim_water_year['yield_m3'].to_numpy()
        self.logger.info(str(self.sim_water_year))
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
        self.params = [spotpy.parameter.Uniform('kc', 0.88, 0.92, 0.01, 0.95),  # crop coefficient
                       spotpy.parameter.Uniform('kr', 90.0, 90.0, 0.001, 0.05),  # vertical conductivity of restrictive layer
                       spotpy.parameter.Uniform('ks', 0.001, 0.001, 0.01, 0.01),  # deep seepage coefficient
                       spotpy.parameter.Uniform('kb', 0.002, 0.002, 0.01, 0.01),  # baseflow coefficient
                       # spotpy.parameter.Uniform('fc', 0.0, 0.8, 0.01, 0.4),  # field capactiy
                       # spotpy.parameter.Uniform('pr', 0.0, 80.0, 0.11, 50.0),  # percent rock
                       ]
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
        gw_coeffs = [200.0, vector[3], vector[2], 1.0001]
        snow_coeffs = [-2.0, 100.0, 250.0]
        soil_prep(self.proj_dir, kr=vector[1], field_cap=None, pct_rock=None)
        result = run_project(self.proj_dir, numcpu=8, gwcoeff=gw_coeffs, pmet=pmet_coeffs, snow=snow_coeffs)
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


class SpotpySetupDaily_ww():
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
        self.params = [spotpy.parameter.Uniform('kc', 0.8, 1.2, 0.01, 0.95),  # crop coefficient
                       spotpy.parameter.Uniform('kb', 0.001, 0.1, 0.01, 0.01),  # baseflow coefficient
                       ]
        self.database = open(os.path.join(self.proj_dir, 'export/calibration_results_annual.csv'), 'w')
        self.countme = 0

    def evaluation(self):
        return self.obs

    def objectivefunction(self, simulation, evaluation):
        # self.logger.info('sim ' + str(simulation))
        # self.logger.info('eval ' + str(evaluation))
        # objectivefunction = spotpy.objectivefunctions.pbias(evaluation, simulation)
        agree = np.where(evaluation == simulation, 1, 0)
        return agree.mean()

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def process_observations(self, obs):
        # self.logger.info(str(obs.head()))
        # evaluation = water_year_yield(obs, conv=3600 * 24 * 0.0283168)  # convert from cfs to cubic meters
        # evaluation = evaluation.loc[(evaluation['year'] >= self.start_year) & (evaluation['year'] <= self.end_year)]
        obs['date_index'] = obs['year'] * 1000 + obs['doy']
        self.df = obs.copy()
        self.chns = obs['chn_id'].unique()
        df_perm = obs.groupby(['chn_id', 'year'], as_index=False).agg({'value': ['count', 'sum']})
        colnames = ['chn_id', 'year', 'obs_count', 'obs_sum']
        df_perm.columns = colnames
        self.df_out = df_perm.copy()
        self.df_match = self.df[['chn_id', 'date_index']]
        evaluation = self.df['value'].to_numpy()
        self.logger.info("EVALUATION SHAPE: " + str(self.df.shape))
        return evaluation

    def save(self, objectivefunctions, parameter, simulations):
        line = str(objectivefunctions)+','+str(parameter).strip('[]')+','+str(simulations.tolist()).strip('[]')+'\n'
        self.logger.info('line' + line)
        self.database.write(line)

    def save_year_counts(self, fn):
        self.df_out.to_csv(fn)

    def simulation(self, vector):
        thresh = 0.0
        self.logger.info('running simulation ' + str(vector))
        pmet_coeffs = [vector[0], 0.8]
        gw_coeffs = [200.0, vector[1], 0.0, 1.0001]
        # pmet_coeffs = [0.95, 0.8]
        # gw_coeffs = [200.0, 0.04, 0.0, 1.0001]
        snow_coeffs = [-2.0, 100.0, 250.0]
        # soil_prep(self.proj_dir, kr=vector[1], field_cap=None, pct_rock=None)
        result = run_project(self.proj_dir, numcpu=8, gwcoeff=gw_coeffs, pmet=pmet_coeffs, snow=snow_coeffs)
        self.logger.info('simulation complete ' + str(result))
        fn_wepp = self.proj_dir + '/wepp/output/chnwb.txt'
        df_wepp = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=25, header=None)
        colnames_units = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=21, header=0, nrows=1)
        df_wepp.columns = colnames_units.columns
        df_wepp['date_index'] = df_wepp['Y'] * 1000 + df_wepp['J']
        # df_wepp = df_wepp.loc[df_wepp['OFE'] == df_wepp['OFE'].max()]
        df_wepp['Qvol'] = (df_wepp['Q'] / 1000.0) * df_wepp['Area']
        df_wepp['Qday'] = (df_wepp['Qvol'] / (3600 * 24))  # cubic meters/second
        df_wepp = df_wepp.copy()
        df_wepp_chns = df_wepp.loc[df_wepp['OFE'].isin(self.chns)].copy()
        dfj = pd.merge(self.df, df_wepp_chns, how='left', left_on=['chn_id', 'date_index'],
                       right_on=['OFE', 'date_index'])
        self.logger.info('SIMULATION SHAPE: ' + str(dfj.shape))
        dfj['wepp_value'] = 0.0
        dfj.loc[dfj['Qday'] > thresh, 'wepp_value'] = 1.0

        # self.logger.info('QDAY MEAN: ' + str(dfj['Qday'].mean()))
        # self.logger.info('DFJ WPP VALUE: ' + str(dfj['wepp_value'].mean()))
        dfj_annual = dfj.groupby(['chn_id', 'year'], as_index=False).agg(
            {'wepp_value': ['sum']})
        # self.logger.info('ANNUAL SHAPE: ' + str(dfj_annual.shape))
        colnames = ['chn_id', 'year', 'wepp_sum']
        dfj_annual.columns = colnames
        dfj_annual = pd.merge(self.df_out, dfj_annual, how="left", left_on=['chn_id', 'year'], right_on=['chn_id', 'year'])
        self.logger.info("OUTPUT SHAPE " + str(self.df_out.shape) + " ADD SHAPE " + str(dfj_annual.shape))
        self.df_out[str(self.countme)] = dfj_annual['wepp_sum'].to_numpy()
        self.logger.info("NEW OUTPUT SHAPE " + str(self.df_out.shape))
        # df_wepp['Qday'] = (df_wepp['Qvol'] / (3600 * 24)) / 0.0283168  # cfs
        # df_mod = water_year_yield(df_wepp, 'date', 'Qvol')
        # df_mod = df_mod.loc[(df_mod['year'] >= self.start_year) & (df_mod['year'] <= self.end_year)]
        # self.logger.info(str(df_mod))

        self.countme += 1
        return dfj['wepp_value'].to_numpy()

class SpotpySetupAnnual_ww():
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
        self.params = [spotpy.parameter.Uniform('kc', 0.8, 1.2, 0.01, 0.95),  # crop coefficient
                       spotpy.parameter.Uniform('kb', 0.001, 0.1, 0.01, 0.01),  # baseflow coefficient
                       ]
        self.database = open(os.path.join(self.proj_dir, 'export/calibration_results_annual.csv'), 'w')

    def evaluation(self):
        return self.obs

    def objectivefunction(self, simulation, evaluation):
        # self.logger.info('sim ' + str(simulation))
        # self.logger.info('eval ' + str(evaluation))
        # objectivefunction = spotpy.objectivefunctions.pbias(evaluation, simulation)
        agree = np.where(evaluation == simulation, 1, 0)
        too_dry = np.where((evaluation == 1) & (simulation == 0), 1, 0)
        too_wet = np.where((evaluation == 0) & (simulation == 1), 1, 0)
        return [agree.mean(), too_dry.mean(), too_wet.mean()]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def process_observations(self, obs):
        # self.logger.info(str(obs.head()))
        # evaluation = water_year_yield(obs, conv=3600 * 24 * 0.0283168)  # convert from cfs to cubic meters
        # evaluation = evaluation.loc[(evaluation['year'] >= self.start_year) & (evaluation['year'] <= self.end_year)]
        obs['date_index'] = obs['year'] * 1000 + obs['doy']
        self.df = obs.copy()
        self.chns = obs['chn_id'].unique()
        df_perm = obs.groupby(['chn_id', 'year'], as_index=False).agg({'value': ['count', 'sum']})
        colnames = ['chn_id', 'year', 'obs_count', 'obs_sum']
        df_perm.columns = colnames
        df_perm['obs_perm'] = 0
        df_perm.loc[df_perm['obs_count'] == df_perm['obs_sum'], 'obs_perm'] = 1
        evaluation = df_perm['obs_perm'].to_numpy()
        self.logger.info("EVALUATION: " + str(evaluation))
        return evaluation

    def save(self, objectivefunctions, parameter, simulations):
        line = str(objectivefunctions)+','+str(parameter).strip('[]')+','+str(simulations.tolist()).strip('[]')+'\n'
        self.logger.info('line' + line)
        self.database.write(line)

    def simulation(self, vector):
        thresh = 0.0
        self.logger.info('running simulation ' + str(vector))
        pmet_coeffs = [vector[0], 0.8]
        gw_coeffs = [200.0, vector[1], 0.0, 1.0001]
        # pmet_coeffs = [0.95, 0.8]
        # gw_coeffs = [200.0, 0.04, 0.0, 1.0001]
        snow_coeffs = [-2.0, 100.0, 250.0]
        # soil_prep(self.proj_dir, kr=vector[1], field_cap=None, pct_rock=None)
        result = run_project(self.proj_dir, numcpu=8, gwcoeff=gw_coeffs, pmet=pmet_coeffs, snow=snow_coeffs)
        self.logger.info('simulation complete ' + str(result))
        fn_wepp = self.proj_dir + '/wepp/output/chnwb.txt'
        df_wepp = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=25, header=None)
        colnames_units = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=21, header=0, nrows=1)
        df_wepp.columns = colnames_units.columns
        df_wepp['date_index'] = df_wepp['Y'] * 1000 + df_wepp['J']
        # df_wepp = df_wepp.loc[df_wepp['OFE'] == df_wepp['OFE'].max()]
        df_wepp['Qvol'] = (df_wepp['Q'] / 1000.0) * df_wepp['Area']
        df_wepp['Qday'] = (df_wepp['Qvol'] / (3600 * 24))  # cubic meters/second
        df_wepp = df_wepp.copy()
        df_wepp_chns = df_wepp.loc[df_wepp['OFE'].isin(self.chns)].copy()
        dfj = pd.merge(self.df, df_wepp_chns, how='left', left_on=['chn_id', 'date_index'],
                       right_on=['OFE', 'date_index'])
        # self.logger.info('JOINED SHAPE: ' + str(dfj.shape))
        dfj['wepp_value'] = 0.0
        dfj.loc[dfj['Qday'] > thresh, 'wepp_value'] = 1.0
        # self.logger.info('QDAY MEAN: ' + str(dfj['Qday'].mean()))
        # self.logger.info('DFJ WPP VALUE: ' + str(dfj['wepp_value'].mean()))
        dfj_annual = dfj.groupby(['chn_id', 'year'], as_index=False).agg(
            {'value': ['count', 'sum'], 'wepp_value': ['count', 'sum']})
        # self.logger.info('ANNUAL SHAPE: ' + str(dfj_annual.shape))
        colnames = ['chn_id', 'year', 'obs_count', 'obs_sum', 'wepp_count', 'wepp_sum']
        dfj_annual.columns = colnames
        dfj_annual['obs_perm'] = 0
        dfj_annual['wepp_perm'] = 0
        dfj_annual.loc[dfj_annual['obs_count'] == dfj_annual['obs_sum'], 'obs_perm'] = 1
        dfj_annual.loc[dfj_annual['wepp_count'] == dfj_annual['wepp_sum'], 'wepp_perm'] = 1
        # self.logger.info('WEPP COUNTS: ' + str(dfj_annual['wepp_sum']))
        dfj_annual['agree'] = 0
        dfj_annual['too_dry'] = 0
        dfj_annual['too_wet'] = 0
        dfj_annual.loc[dfj_annual['obs_perm'] == dfj_annual['wepp_perm'], 'agree'] = 1
        dfj_annual.loc[(dfj_annual['obs_perm'] == 1) & (dfj_annual['wepp_perm'] == 0), 'too_dry'] = 1
        dfj_annual.loc[(dfj_annual['obs_perm'] == 0) & (dfj_annual['wepp_perm'] == 1), 'too_wet'] = 1
        # df_wepp['Qday'] = (df_wepp['Qvol'] / (3600 * 24)) / 0.0283168  # cfs
        # df_mod = water_year_yield(df_wepp, 'date', 'Qvol')
        # df_mod = df_mod.loc[(df_mod['year'] >= self.start_year) & (df_mod['year'] <= self.end_year)]
        # self.logger.info(str(df_mod))

        return dfj_annual['wepp_perm'].to_numpy()