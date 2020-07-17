import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from run_project import *


def nse(obs, sim):
    top = np.sum((sim - obs) ** 2)
    bottom = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - (top / bottom)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('wd', type=str,
                        help='path of project')

    gw_params = [[200, 0.07, 0.01, 1],
                 [200, 0.07, 0.1, 1],
                 [200, 0.07, 0, 1],
                 [200, 0.06, 0.01, 1],
                 [200, 0.06, 0.1, 1],
                 [200, 0.06, 0, 1],
                 [200, 0.05, 0.01, 1],
                 [200, 0.05, 0.1, 1],
                 [200, 0.05, 0, 1],
                 [200, 0.04, 0.01, 1],
                 [200, 0.04, 0.1, 1],
                 [200, 0.04, 0, 1],
                 [200, 0.03, 0.01, 1],
                 [200, 0.03, 0.1, 1],
                 [200, 0.03, 0, 1]]

    args = parser.parse_args()

    wd = args.wd

    fn_hist = r'C:\konrad\projects\usgs\hjandrews\data\discharge\HF00402_v12.csv'  # daily discharge for all gaged watersheds
    ws_code = 'GSWS01'  # watershed of interest
    datetime_format = '%Y-%m-%d'
    start_date = '2012-01-01'
    end_date = '2015-12-31'
    df_hist = pd.read_csv(fn_hist)
    df_hist['DATE'] = pd.to_datetime(df_hist['DATE'], format=datetime_format)  # convert dates to date type
    df_hist = df_hist.loc[df_hist['SITECODE'] == ws_code]  # subset to watershed of interest
    df_hist = df_hist.loc[(df_hist['DATE'] >= start_date) & (df_hist['DATE'] <= end_date)]

    for params in gw_params:
        print(params)
        run_project(wd, numcpu=8, gwcoeff=params)
        fn_wepp = wd + '/wepp/output/chnwb.txt'

        df_wepp = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=25, header=None)
        colnames_units = pd.read_table(fn_wepp, delim_whitespace=True, skiprows=21, header=0, nrows=1)
        df_wepp.columns = colnames_units.columns
        df_wepp['Qvol'] = (df_wepp['Q'] / 1000.0) * df_wepp['Area']
        df_wepp['Qday'] = (df_wepp['Qvol'] / (3600 * 24)) / 0.0283168  # cfs
        print('WEPP metrics:', df_wepp['Qday'].min(), df_wepp['Qday'].max(), df_wepp['Qday'].mean())

        df_wepp = df_wepp.loc[df_wepp['Y'] > 2011]
        df_wepp = df_wepp.loc[df_wepp['OFE'] == 19]
        nse_val = nse(df_hist['MEAN_Q'].values, df_wepp['Qday'].values)
        params.append(nse_val)

        plt.figure()
        plt.plot(df_hist['DATE'], df_hist['MEAN_Q'])
        plt.plot(df_hist['DATE'], df_wepp['Qday'], alpha=0.5)
        plot_text = "BF Storage: {bfs}\n" \
                    "BF Recession Coef (k): {bfk}\n" \
                    "Deep Seepage: {bfp}\n" \
                    "BF Area: {bfa}\n" \
                    "NSE: {nse}".format(bfs=params[0], bfk=params[1], bfp=params[2], bfa=params[3], nse=params[4])
        plt.text(0.7, 0.75, plot_text, transform=plt.gca().transAxes)
        plt.savefig(wd + "/export/calibration_plots/result{bfs}-{bfk}-{bfp}-{bfa}.png".format(bfs=params[0],
                                                                                              bfk=params[1],
                                                                                              bfp=params[2],
                                                                                              bfa=params[3]))

        print('NSE', nse_val)
    df_out = pd.DataFrame(gw_params, columns=['storage', 'bk', 'ds', 'area', 'nse'])
    df_out.to_csv(wd + "/export/calibration_results.csv")
    print(df_out)