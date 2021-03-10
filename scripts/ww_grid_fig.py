import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor


proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
results_fn = "export/calibration_results_daily_perm.npy"
eval_fn = "export/calibration_results_daily_perm_eval.npy"
df_fn = "export/calibration_results_daily_perm_annual_counts.csv"
in_obs = r"E:\konrad\Projects\usgs\hjandrews\data\thermister\working\ww_thermister_observations_wepp.csv"
df_obs = pd.read_csv(in_obs)

nparam = 2
nlike = 1
n_pre_col = nparam + nlike
day_thresh = 5  # number of days required for a stream to be non-permanent

best_daily_params = [6, 29, 59, 79]  # from ww_daily_analysis script
best_annual_params = [91, 54, 76, 93]  # row of best params for annual accuracy from ww_annual_analysis script
param_set = best_annual_params

proj_names = ['ww-ws1-base', 'ww-ws2-base', 'ww-ws3-base', 'ww-ws4-base']

for i in range(len(proj_names)):
    ws_df = df_obs.loc[df_obs['wshed'] == i + 1]
    doy = ws_df['doy'].to_numpy()
    # subset to the portion of the year from April - October (inclusive)
    date_index = np.where((doy > 90) & (doy < 304))
    ws_df = ws_df.loc[(df_obs['doy'] > 90) & (ws_df['doy'] < 304)]
    print('df shape', ws_df.shape)
    # get simulation and evaluation results
    simulation = np.asarray(np.load(os.path.join(proj_base, proj_names[i], results_fn)).tolist())
    evaluation = np.asarray(np.load(os.path.join(proj_base, proj_names[i], eval_fn)).tolist())
    # df = pd.read_csv(os.path.join(proj_base, proj_names[i], df_fn))
    # subset evaluation data
    evaluation = evaluation[date_index[0]]
    # get only simulation results (remove likelihood, parameter values, and chain number
    sims = simulation[:, n_pre_col:-1]
    # subset simulations to date range
    sims = sims[:, date_index[0]]
    ws_df['eval'] = evaluation
    colnames = ['chn_id', 'year', 'eval']
    count_df = ws_df[colnames]
    count_df = count_df.groupby(['chn_id', 'year'], as_index=False).count()

    for row in range(sims.shape[0]):
        ws_df[str(row)] = sims[row, :]
        colnames.append(str(row))

    ws_df = ws_df[colnames]
    ws_df = ws_df.groupby(['chn_id', 'year'], as_index=False).sum()
    ws_df['total'] = count_df['eval'].to_numpy()
    ws_df['perm'] = 0
    ws_df.loc[ws_df['total'] == ws_df['eval'], 'perm'] = 1
    daily_df = ws_df[['chn_id', 'year', 'perm', 'total', str(param_set[i])]]
    pt = pd.pivot_table(ws_df, values='eval', index='chn_id', columns='year')
    eval_array = pt.to_numpy()
    pt = pd.pivot_table(ws_df, values='total', index='chn_id', columns='year')
    total_array = pt.to_numpy()  # total number of observations for each reach in each year
    pt = pd.pivot_table(ws_df, values=str(param_set[i]), index='chn_id', columns='year')
    count_array = pt.to_numpy()  # number of days modeled wet
    pt = pd.pivot_table(ws_df, values='perm', index='chn_id', columns='year')
    perm_array = pt.to_numpy()  # permanence classification from thermistor data
    plot_array = np.zeros(perm_array.shape)  # array to use for plot
    plot_array = np.where(perm_array == 0, -1, plot_array)  # non perm get value of -1
    plot_array = np.where(perm_array == 1, 1, plot_array)  # perm get value of 1
    plot_array = np.where((perm_array == 1) & (count_array < (total_array-day_thresh)), -2, plot_array)  # if perm modeled as non perm get value -2
    plot_array = np.where((perm_array == 0) & (count_array >= (total_array-day_thresh)), 2, plot_array)  # if non perm modeled as perm get value 2
    value_array = 1.0 - (np.fabs(count_array - eval_array) * 1.0 / total_array * 1.0)

    chns = ws_df['chn_id'].unique()
    years = ws_df['year'].unique()

    plt.figure(figsize=(4, 4))
    plt.imshow(plot_array, aspect='auto', interpolation='none', cmap='bwr_r', vmin=-2, vmax=2)
    plt.yticks(np.arange(len(chns)), chns.astype(np.int))
    plt.xticks(np.arange(len(years)), years)

    for row in range(len(chns)):
        for col in range(len(years)):
            if plot_array[row, col] != 0:
                text = plt.text(col, row, round_decimals_down(value_array[row, col], 2), ha="center", va="center", color="w", fontweight='bold', fontsize='10')


    plt.imshow(np.full(perm_array.shape, 1), aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1, alpha=0.2)
    for chn in range(len(chns)):
        plt.axhline(y=chn - 0.5, color='k')
    for yr in range(len(years)):
        plt.vlines(yr - 0.5, -0.5, len(chns) - 0.5, color='k')
    plt.subplots_adjust(left=0.1, right=0.97, top=0.97)
    plt.xlabel('Year')
    plt.show()