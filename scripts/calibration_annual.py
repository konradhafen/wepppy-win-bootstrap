import argparse
import pandas as pd
import spotpy
from spotpy_setup import *


def water_year_yield(df):

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
    start_year = args.start_year
    end_year = args.end_year
    output = wd + '/export/calibration_results_withks.csv'
    # start_date = '2012-01-01'
    # end_date = '2015-12-31'
    df_hist = pd.read_csv(fn_hist)
    df_hist['DATE'] = pd.to_datetime(df_hist['DATE'], format=datetime_format)  # convert dates to date type
    df_hist = df_hist.loc[df_hist['SITECODE'] == ws_code]  # subset to watershed of interest
    df_hist = water_year_yield(df_hist)

    spot_setup = SpotpySetupAnnual(wd, start_year, end_year, df_hist)
    sampler = spotpy.algorithms.mc(spot_setup, dbname=output, dbformat='csv')
    sampler.sample(args.reps)