import pandas as pd
import os
import argparse
from spotpy_setup import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('reps', type=int, help='number of repetitions')
    parser.add_argument('wd', type=str, help='path to wepp project')
    parser.add_argument('start_date', type=str, help='start date')
    parser.add_argument('end_date', type=str, help='end date')
    parser.add_argument('site_code', type=int, help='id of watershed with observed data')
    # parser.add_argument('datetime_format', type=str, help='format for start and end dates, default: %Y-%m-%d')

    args = parser.parse_args()
    wd = args.wd
    ws_code = args.site_code
    # fn_hist = r'C:\konrad\projects\usgs\hjandrews\data\discharge\HF00402_v12.csv'  # daily discharge for all gaged watersheds
    fn_hist = r'E:\konrad\Projects\usgs\hjandrews\data\discharge\HF00402_v12.csv'
    in_obs = r"E:\konrad\Projects\usgs\hjandrews\data\thermister\working\ww_thermister_observations_wepp.csv"
    df_obs = pd.read_csv(in_obs)
    df_obs = df_obs.loc[df_obs['wshed'] == ws_code]
    # ws_code = 'GSWS01'  # watershed of interest
    datetime_format = '%Y-%m-%d'
    start_date = args.start_date
    end_date = args.end_date
    output = os.path.join(wd, 'export/calibration_results_daily_perm')
    print("wd, output", wd, output)
    # start_date = '2012-01-01'
    # end_date = '2015-12-31'

    spot_setup = SpotpySetupDaily_ww(wd, start_date, end_date, df_obs)
    sampler = spotpy.algorithms.mc(spot_setup, dbname=output, dbformat='csv')
    sampler.sample(args.reps)
    results = sampler.getdata()
    spot_setup.save_year_counts(output + "_annual_counts.csv")
    if os.path.exists(output + '.npy'):
        os.remove(output + '.npy')
    np.save(output + '.npy', results)
    if os.path.exists(output + '_eval.npy'):
        os.remove(output + '_eval.npy')
    np.save(output + '_eval.npy', spot_setup.evaluation())