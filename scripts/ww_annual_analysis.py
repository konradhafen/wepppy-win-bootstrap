import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


########################################################################################################################
# This section calculated annual permanence accuracy based on results from daily runs
########################################################################################################################
proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
results_fn = "export/calibration_results_daily_perm.npy"
eval_fn = "export/calibration_results_daily_perm_eval.npy"
df_fn = "export/calibration_results_daily_perm_annual_counts.csv"
in_obs = r"E:\konrad\Projects\usgs\hjandrews\data\thermister\working\ww_thermister_observations_wepp.csv"
df_obs = pd.read_csv(in_obs)

nparam = 2
nlike = 1
n_pre_col = nparam + nlike

proj_names = ['ww-ws1-base', 'ww-ws2-base', 'ww-ws3-base', 'ww-ws4-base']
# proj_names = ['ww-ws1-base', 'ww-ws2-base', 'ww-ws3-base']

fig, axs = plt.subplots(len(proj_names), sharex=True, sharey=True, figsize=(5, 7))
pt_size = 20.0
alpha = 0.5
accuracy_index = None
param_row = np.empty(len(proj_names))

for i in range(len(proj_names)):
    ws_df = df_obs.loc[df_obs['wshed'] == i+1]
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
    print(count_df.head())
    for row in range(sims.shape[0]):
        ws_df[str(row)] = sims[row, :]
        colnames.append(str(row))
    # print('updated df columns', ws_df.columns)
    ws_df = ws_df[colnames]
    ws_df = ws_df.groupby(['chn_id', 'year'], as_index=False).sum()
    ws_df['perm'] = np.where(count_df['eval'].to_numpy() == ws_df['eval'].to_numpy(), 1, 0)
    print(ws_df.head())

########################################################################################################################
# Section below is for use with WEPP runs that specifically calculated annual permanence
########################################################################################################################

# proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
# results_fn = "export/calibration_results_annual_perm.npy"
# eval_fn = "export/calibration_results_annual_perm_eval.npy"
#
# proj_names = ['ww-ws1-base', 'ww-ws2-base', 'ww-ws3-base', 'ww-ws4-base']

# fig, axs = plt.subplots(len(proj_names), sharex=True, sharey=True, figsize=(5, 7))
# n_pre_col = 5
# alpha = 0.5
# pt_size = 20.0
#
# for i in range(len(proj_names)):
#     simulation = np.asarray(np.load(os.path.join(proj_base, proj_names[i], results_fn)).tolist())
#     evaluation = np.asarray(np.load(os.path.join(proj_base, proj_names[i], eval_fn)).tolist())
#     accuracy = np.zeros((simulation.shape[0], 3))
#     accuracy[:, 0] = simulation[:, 0]
#     eval_dry = np.where(evaluation == 0)
#     eval_wet = np.where(evaluation == 1)
#     sims = simulation[:, n_pre_col:-1]
#     sims_dry = sims[:, eval_dry[0]]
#     accuracy[:, 1] = 1.0 - sims_dry.sum(axis=1)/len(eval_dry[0])
#     sims_wet = sims[:, eval_wet[0]]
#     accuracy[:, 2] = sims_wet.sum(axis=1)/len(eval_wet[0])
#     print(proj_names[i])
#     print(np.unique(accuracy, axis=0))
#     plot_dat = np.unique(accuracy, axis=0)
#     axs[i].scatter(plot_dat[:, 0], plot_dat[:, 1], color='r', alpha=alpha, s=pt_size, label="Non-permanent Accuracy")
#     axs[i].scatter(plot_dat[:, 0], plot_dat[:, 2], color='b', alpha=alpha, s=pt_size, label="Permanent Accuracy")
#     axs[i].scatter(plot_dat[:, 0], plot_dat[:, 0] - np.fabs(plot_dat[:, 1] - plot_dat[:, 2]), color='k', s=pt_size, label="Adjusted Accuracy")
#     axs[i].set_title('Willow-Whitehorse ' + str(i+1).zfill(2) + ' (n=' + str(sims.shape[1]) + ")", fontsize=10)
#
# fig.subplots_adjust(bottom=0.15, hspace=0.4, top=0.95, right=0.95, left=0.1)
# plt.xlabel('Overall Accuracy')
# plt.xlim((0.0, 1.0))
# plt.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -1.0))
# plt.show()