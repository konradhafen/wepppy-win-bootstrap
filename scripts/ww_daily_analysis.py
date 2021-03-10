import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


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

    if accuracy_index is None:
        # create array for adjusted accuracy values
        accuracy_index = np.empty((sims.shape[0], len(proj_names)))
    # empty array for accuracy calculations
    accuracy = np.zeros((simulation.shape[0], 3))
    # number of observed wet and dry days
    eval_dry = np.where(evaluation == 0)
    eval_wet = np.where(evaluation == 1)
    # simulated value on dry days
    sims_dry = sims[:, eval_dry[0]]
    # accuracy of dry days
    accuracy[:, 1] = 1.0 - sims_dry.sum(axis=1) / len(eval_dry[0])
    # simulated value on wet days
    sims_wet = sims[:, eval_wet[0]]
    # accuracy of wet days
    accuracy[:, 2] = sims_wet.sum(axis=1) / len(eval_wet[0])
    # overall accuracy
    accuracy[:, 0] = (sims_wet.sum(axis=1) + (len(eval_dry[0]) - sims_dry.sum(axis=1))) / sims.shape[1]
    # adjusted accuracy
    accuracy_index[:, i] = accuracy[:, 0] - np.fabs(accuracy[:, 1] - accuracy[:, 2])
    # removing duplicate entries for plotting
    plot_dat = np.unique(accuracy, axis=0)
    axs[i].scatter(plot_dat[:, 0], plot_dat[:, 1], color='r', alpha=alpha, s=pt_size, label="Dry Accuracy")
    axs[i].scatter(plot_dat[:, 0], plot_dat[:, 2], color='b', alpha=alpha, s=pt_size, label="Wet Accuracy")
    axs[i].scatter(plot_dat[:, 0], plot_dat[:, 0] - np.fabs(plot_dat[:, 1] - plot_dat[:, 2]), color='k', s=pt_size,
                   label="Adjusted Accuracy")
    # axs[i].scatter(plot_dat[:, 0], plot_dat[:, 1] * plot_dat[:, 2], color='g', alpha=alpha, s=pt_size, label="Multiplied")
    axs[i].set_title('Willow-Whitehorse ' + str(i + 1).zfill(2) + ' (n=' + str(sims.shape[1]) + ")", fontsize=10)
    param_row[i] = np.where(accuracy_index[:, i] == np.max(accuracy_index[:, i]))[0][0]
    print('Row(s) of best parameter set', param_row[i], accuracy[int(param_row[i]), :])

    # accuracy_index[np.where(accuracy_index[:, i] == np.max(accuracy_index[:, i])), i]

fig.subplots_adjust(bottom=0.15, hspace=0.4, top=0.95, right=0.95, left=0.1)
plt.xlabel('Overall Accuracy')
plt.xlim((0.25, 1.0))
plt.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -1.0))
plt.show()

# for i in range(len(proj_names)):
#     df = pd.read_csv(os.path.join(proj_base, proj_names[i], df_fn))
#     # print(df[['obs_count', 'obs_sum', str(int(param_row[i]))]])
#     # print('years', df['year'].unique().shape)
#     sims = df.iloc[:, -100:].to_numpy()
#     counts = df.iloc[:, 4].to_numpy()
#     # print(df.columns[:6])
#     # print('sim shape', sims.shape, 'counts type', type(counts), 'counts shape', counts.shape)
#     sims_sub = (sims.transpose() - counts).transpose()
#     maxv = np.fabs(sims_sub).max()
#
#     df.iloc[:, -100:] = sims_sub
#     pt = pd.pivot_table(df, values=str(int(param_row[i])), index='chn_id', columns='year')
#     fig_array = pt.to_numpy()
#     chns = df['chn_id'].unique()
#     years = df['year'].unique()
#
#     df['perm'] = 0
#     df['mod_perm'] = 0
#     df.loc[df['obs_count'] == df['obs_sum'], 'perm'] = 1
#     df.loc[df['obs_count'] == df[str(int(param_row[i]))], 'mod_perm'] = 1
#     df['perm_dif'] = 0
#     df.loc[(df['perm'] == 0) & (df['mod_perm'] == 1), 'perm_dif'] = 1
#     df.loc[(df['perm'] == 1) & (df['mod_perm'] == 0), 'perm_dif'] = -1
#     pt = pd.pivot_table(df, values='perm', index='chn_id', columns='year')
#     perm_array = pt.to_numpy()
#     pt = pd.pivot_table(df, values='perm_dif', index='chn_id', columns='year')
#     change_array = pt.to_numpy()
#     plt.figure(figsize=(4, 4))
#     # plt.imshow(fig_array, aspect='auto', interpolation='none', cmap='RdBu', vmin=maxv*(-1.0), vmax=maxv)
#     plt.imshow(change_array, aspect='auto', interpolation='none', cmap='bwr_r', vmin=-1, vmax=1)
#     plt.yticks(np.arange(len(chns)), chns.astype(np.int))
#     plt.xticks(np.arange(len(years)), years)
#
#     for row in range(len(chns)):
#         for col in range(len(years)):
#             if change_array[row, col] >= -1:
#                 text = plt.text(col, row, np.abs(fig_array[row, col].astype(int)), ha="center", va="center", color="w", fontweight='bold', fontsize='14')
#             # elif perm_array[row, col] == 0:
#             #     text = plt.text(col, row, np.abs(fig_array[row, col].astype(int)), ha="center", va="center", color="k", fontweight='bold', fontsize='14')
#
#     np.where(perm_array == 0, np.nan, 1)
#
#     plt.imshow(np.full(perm_array.shape, 1), aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1, alpha=0.2)
#             # if fig_array[row, col] < -366:
#             #     text = plt.text(col, row, "", ha="center", va="center", color="w")
#     # for chn in chns[:-1]:
#     #     plt.axhline(y=chn - 0.5, color='k')
#     # plt.yticks(chns - 1, chns)
#     # plt.ylabel('Channel ID')
#     # plt.xlabel('Parameter Set')
#     # plt.title('Willow-Whitehorse ' + str(i+1).zfill(2))
#     # plt.colorbar()
#     # plt.subplots_adjust(left=0.05, right=0.99)
#     # plt.savefig(os.path.join(fig_base, labels[i] + ".png"), dpi=300)
#     plt.show()
