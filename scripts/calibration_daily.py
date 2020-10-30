import argparse
import pandas as pd
import numpy as np
import os
import spotpy
from spotpy_setup import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('reps', type=int, help='number of repetitions')
    parser.add_argument('wd', type=str, help='path to wepp project')
    parser.add_argument('start_date', type=str, help='start date')
    parser.add_argument('end_date', type=str, help='end date')
    parser.add_argument('site_code', type=str, help='id of watershed with observed data')
    # parser.add_argument('datetime_format', type=str, help='format for start and end dates, default: %Y-%m-%d')

    args = parser.parse_args()
    wd = args.wd
    # fn_hist = r'C:\konrad\projects\usgs\hjandrews\data\discharge\HF00402_v12.csv'  # daily discharge for all gaged watersheds
    fn_hist = r'E:\konrad\Projects\usgs\hjandrews\data\discharge\HF00402_v12.csv'
    # ws_code = 'GSWS01'  # watershed of interest
    ws_code = args.site_code
    datetime_format = '%Y-%m-%d'
    start_date = args.start_date
    end_date = args.end_date
    output = os.path.join(wd, 'export/calibration_results_daily')
    print("wd, output", wd, output)
    # start_date = '2012-01-01'
    # end_date = '2015-12-31'
    df_hist = pd.read_csv(fn_hist)
    df_hist['DATE'] = pd.to_datetime(df_hist['DATE'], format=datetime_format)  # convert dates to date type
    df_hist = df_hist.loc[df_hist['SITECODE'] == ws_code]  # subset to watershed of interest

    spot_setup = SpotpySetup(wd, start_date, end_date, df_hist)
    sampler = spotpy.algorithms.mc(spot_setup, dbname=output + '.csv', dbformat='csv')
    sampler.sample(args.reps)
    results = sampler.getdata()
    np.save(output + '.npy', results)
    np.save(output + '_eval.npy', spot_setup.evaluation())