import argparse
import pandas as pd
import numpy as np
import spotpy
from spotpy_setup import *


def pbias(sim, obs):
    return 100 * (np.sum(sim-obs) / np.sum(obs))


def water_year_yield(df, date_col='DATE', q_col='MEAN_Q', conv=1.0):
    years = pd.DatetimeIndex(df[date_col]).year.unique()
    vol = []
    for i in range(len(years) - 1):
        start_date = str(years[i]) + '-10-01'
        end_date = str(years[i+1]) + '-09-30'
        tmp_dat = df.loc[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
        vol.append(tmp_dat[q_col].sum() * conv)  # sum and convert units
        # print(years[i], tmp_dat.shape, vol[-1])

    df_wy = pd.DataFrame({'year': years[:-1], 'yield_m3': vol})
    return df_wy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('reps', type=int, help='number of repetitions')
    parser.add_argument('wd', type=str, help='path to wepp project')
    parser.add_argument('start_year', type=str, help='start water year')
    parser.add_argument('end_year', type=str, help='end water year')
    parser.add_argument('site_code', type=str, help='id of watershed with observed data')
    # parser.add_argument('datetime_format', type=str, help='format for start and end dates, default: %Y-%m-%d')

    args = parser.parse_args()
    wd = args.wd
    # fn_hist = r'C:\konrad\projects\usgs\hjandrews\data\discharge\HF00402_v12.csv'  # daily discharge for all gaged watersheds
    fn_hist = r'E:\konrad\Projects\usgs\hjandrews\data\discharge\HF00402_v12.csv'
    # ws_code = 'GSWS01'  # watershed of interest
    ws_code = args.site_code
    datetime_format = '%Y-%m-%d'
    start_year = int(args.start_year)
    end_year = int(args.end_year)
    output = wd + '/export/calibration_results_annual.csv'
    # start_date = '2012-01-01'
    # end_date = '2015-12-31'
    df_hist = pd.read_csv(fn_hist)
    df_hist['DATE'] = pd.to_datetime(df_hist['DATE'], format=datetime_format)  # convert dates to date type
    df_hist = df_hist.loc[df_hist['SITECODE'] == ws_code]  # subset to watershed of interest
    spot_setup = SpotpySetupAnnual(wd, 2005, 2015, df_hist)
    sampler = spotpy.algorithms.mc(spot_setup, dbname=output, dbformat='csv')
    sampler.sample(args.reps)



########################################################################################################################
# Testing pbias and making sure things look realistic
########################################################################################################################


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('reps', type=int, help='number of repetitions')
#     parser.add_argument('wd', type=str, help='path to wepp project')
#     parser.add_argument('start_year', type=str, help='start water year')
#     parser.add_argument('end_year', type=str, help='end water year')
#     parser.add_argument('site_code', type=str, help='id of watershed with observed data')
#     # parser.add_argument('datetime_format', type=str, help='format for start and end dates, default: %Y-%m-%d')
#
#     args = parser.parse_args()
#     wd = args.wd
#     # fn_hist = r'C:\konrad\projects\usgs\hjandrews\data\discharge\HF00402_v12.csv'  # daily discharge for all gaged watersheds
#     fn_hist = r'E:\konrad\Projects\usgs\hjandrews\data\discharge\HF00402_v12.csv'
#     # ws_code = 'GSWS01'  # watershed of interest
#     ws_code = args.site_code
#     datetime_format = '%Y-%m-%d'
#     start_year = int(args.start_year)
#     end_year = int(args.end_year)
#     output = wd + '/export/calibration_results_annual.csv'
#     # start_date = '2012-01-01'
#     # end_date = '2015-12-31'
#     df_hist = pd.read_csv(fn_hist)
#     df_hist['DATE'] = pd.to_datetime(df_hist['DATE'], format=datetime_format)  # convert dates to date type
#     df_hist = df_hist.loc[df_hist['SITECODE'] == ws_code]  # subset to watershed of interest
#     df_hist_q = df_hist.copy()
#     df_hist = water_year_yield(df_hist, conv=3600 * 24 * 0.0283168)
#
#     fn_wepp = wd + '/wepp/output/chnwb.txt'
#     df_wepp = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=25, header=None)
#     colnames_units = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=21, header=0, nrows=1)
#     df_wepp.columns = colnames_units.columns
#     df_wepp = df_wepp.loc[df_wepp['OFE'] == df_wepp['OFE'].max()]
#     df_wepp['date'] = pd.to_datetime(df_wepp['Y'] * 1000 + df_wepp['J'], format='%Y%j')
#     # print(df_wepp['date'])
#     # df_wepp['date'] = df_wepp['date'].dt.strftime(datetime_format)
#     # print(df_wepp['date'])
#     df_wepp['Qvol'] = (df_wepp['Q'] / 1000.0) * df_wepp['Area']
#     df_wepp['Qday'] = (df_wepp['Qvol'] / (3600 * 24)) / 0.0283168  # cfs
#     print('mean daily Q from WEPP')
#     # print(df_wepp['Qday'])
#     df_mod = water_year_yield(df_wepp, 'date', 'Qvol')
#
#     df_hist = df_hist.loc[(df_hist['year'] >= start_year) & (df_hist['year'] <= end_year)]
#     df_mod = df_mod.loc[(df_mod['year'] >= start_year) & (df_mod['year'] <= end_year)]
#
#     # print('observations')
#     # print(df_hist)
#     # print('simulations')
#     # print(df_mod)
#     print(np.corrcoef(df_hist['yield_m3'].to_numpy(), df_mod['yield_m3'].to_numpy()))
#     print('annual pbias')
#     print(pbias(df_mod['yield_m3'].to_numpy(), df_hist['yield_m3'].to_numpy()))
#
#     df_hist_q['year'] = pd.DatetimeIndex(df_hist_q['DATE']).year
#     df_hist_q = df_hist_q.loc[(df_hist_q['year'] >= start_year) & (df_hist_q['year'] <= end_year)]
#     df_wepp = df_wepp.loc[(df_wepp['Y'] >= start_year) & (df_wepp['Y'] <= end_year)]
#     print('daily pbias')
#     print(pbias(df_wepp['Qday'].to_numpy(), df_hist_q['MEAN_Q'].to_numpy()))
#
#     # spot_setup = SpotpySetupAnnual(wd, start_year, end_year, df_hist)
#     # sampler = spotpy.algorithms.mc(spot_setup, dbname=output, dbformat='csv')
#     # sampler.sample(args.reps)