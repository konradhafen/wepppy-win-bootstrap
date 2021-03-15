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
df = None

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
n_threshold_days = 21
thresh_array = np.full((n_threshold_days*4, 7), -1.0, dtype=np.float)
best_threshold_days = [0, 8, 3, 2]  # number of theshold days for best accuracy, identified from the graph created running this script
best_param_rows_thresh = [91, 14, 11, 93]  # index of best params for corresponding threshold days, idnetified from the variable above
best_param_rows = [91, 54, 76, 93]

for i in range(len(proj_names)):
    ws_df = df_obs.loc[df_obs['wshed'] == i+1]
    doy = ws_df['doy'].to_numpy()
    # subset to the portion of the year from April - October (inclusive)
    date_index = np.where((doy > 90) & (doy < 304))
    ws_df = ws_df.loc[(df_obs['doy'] > 90) & (ws_df['doy'] < 304)]
    # print('df shape', ws_df.shape)
    # get simulation and evaluation results
    simulation = np.asarray(np.load(os.path.join(proj_base, proj_names[i], results_fn)).tolist())
    evaluation = np.asarray(np.load(os.path.join(proj_base, proj_names[i], eval_fn)).tolist())
    print('best param values for WW', i)
    print(simulation[best_param_rows_thresh[i], :n_pre_col])
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
    pt = pd.pivot_table(ws_df, values=str('eval'), columns='year', index='chn_id')
    obs_array = pt.to_numpy()  # sum of wet observations on each channel
    pt = pd.pivot_table(count_df, values='eval', index='chn_id', columns='year')
    total_array = pt.to_numpy()  # total number of observations on each channel
    perm_array = np.where(obs_array == total_array, 1, -1)
    perm_array = np.where(obs_array < total_array, 0, perm_array)
    wet_count = ws_df['eval'].to_numpy()
    total_count = count_df['eval'].to_numpy()
    perm = np.where(total_count == wet_count, 1, 0)
    error_df = ws_df[['chn_id', 'year']].copy()

    # ws_df['perm'] = np.where(count_df['eval'].to_numpy() == ws_df['eval'].to_numpy(), 1, 0)
    # ws_df['total'] = count_df['eval']
    accuracy = np.empty((sims.shape[0], 5))
    for row in range(sims.shape[0]):
        pt = pd.pivot_table(ws_df, values=str(row), columns='year', index='chn_id')
        count_array = pt.to_numpy()

        for thresh in range(n_threshold_days):
            tv = thresh + i * n_threshold_days
            mod_thresh_array = np.where(count_array >= (total_array - thresh), 1, 0)
            oa = np.where(mod_thresh_array == perm_array, 1, 0).sum() / np.where(perm > -1, 1, 0).sum()
            wa = np.where(mod_thresh_array[np.where(perm_array == 1)] == perm_array[np.where(perm_array == 1)], 1, 0).sum() / np.where(perm == 1, 1, 0).sum()
            da = np.where(mod_thresh_array[np.where(perm_array == 0)] == perm_array[np.where(perm_array == 0)], 1, 0).sum() / np.where(perm == 0, 1, 0).sum()

            if (oa - np.fabs(wa - da)) > thresh_array[tv, 3]:
                # print('updating values')
                thresh_array[tv, 0] = oa
                thresh_array[tv, 1] = wa
                thresh_array[tv, 2] = da
                thresh_array[tv, 3] = thresh_array[tv, 0] - np.fabs(thresh_array[tv, 1] - thresh_array[tv, 2])
                thresh_array[tv, 4] = row
                thresh_array[tv, 5] = i
                thresh_array[tv, 6] = thresh
                # print('threshold values updated', thresh_array[tv, :])

        mod_array = np.where(count_array == total_array, 1, 0)

        accuracy[row, 0] = np.where(mod_array == perm_array, 1, 0).sum() / np.where(perm > -1, 1, 0).sum()  # Overall accuracy
        accuracy[row, 1] = np.where(mod_array[np.where(perm_array == 1)] == perm_array[np.where(perm_array == 1)], 1, 0).sum() / np.where(perm == 1, 1, 0).sum()  # Permanent accuracy
        accuracy[row, 2] = np.where(mod_array[np.where(perm_array == 0)] == perm_array[np.where(perm_array == 0)], 1, 0).sum() / np.where(perm == 0, 1, 0).sum()   # Non-permanent accuracy
        accuracy[row, 4] = (accuracy[row, 1] * 0.5) + (accuracy[row, 2] * 0.5)
        accuracy[row, 3] = accuracy[row, 0] - np.fabs(accuracy[row, 1] - accuracy[row, 2])
    # print(error_df.head())
    # print("count_array")
    # print(count_array)
    plot_dat = np.unique(accuracy, axis=0)
    axs[i].scatter(plot_dat[:, 0], plot_dat[:, 1], color='b', alpha=alpha, s=pt_size, label="Permanent Accuracy")
    axs[i].scatter(plot_dat[:, 0], plot_dat[:, 2], color='r', alpha=alpha, s=pt_size, label="Non-permanent Accuracy")
    axs[i].scatter(plot_dat[:, 0], plot_dat[:, 3], color='k', s=pt_size, label="Adjusted Accuracy")
    axs[i].set_ylabel('Accuracy Value')
    # axs[i].scatter(plot_dat[:, 0], plot_dat[:, 4], color='g', s=pt_size, label="Weigthed Accuracy")
    # axs[i].scatter(plot_dat[:, 0], (plot_dat[:, 4] + plot_dat[:, 3]) / 2.0, color='y', s=pt_size, label="Accuracy Index")
    axs[i].set_title('Willow-Whitehorse ' + str(i + 1).zfill(2) + ' (n=' + str(ws_df.shape[0]) + ")", fontsize=10)
    br = np.where(accuracy[:, 3] == np.max(accuracy[:, 3]))[0][0]
    print('Row of best parameter set', br, accuracy[br, :])
    # axs[i].scatter(plot_dat[:, 0], plot_dat[:, 0] - np.fabs(plot_dat[:, 1] - plot_dat[:, 2]), color='k', s=pt_size, label="Adjusted Accuracy")
    # axs[i].scatter(plot_dat[:, 0], plot_dat[:, 1] * plot_dat[:, 2], color='g', alpha=alpha, s=pt_size, label="Multiplied")
fig.subplots_adjust(bottom=0.15, hspace=0.4, top=0.95, right=0.95, left=0.13)
plt.xlabel('Overall Accuracy')
plt.setp(axs, xlim=(0.0, 1.0))
syms, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(syms, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
plt.show()

fig, axs = plt.subplots(len(proj_names), sharex=True, sharey=True, figsize=(5, 7))
for i in range(len(proj_names)):
    plot_dat = thresh_array[(i*n_threshold_days):(i*n_threshold_days + n_threshold_days), :]
    axs[i].scatter(plot_dat[:, 6].astype(np.int), plot_dat[:, 1], color='b', alpha=alpha, s=pt_size, label="Permanent Accuracy")
    axs[i].scatter(plot_dat[:, 6].astype(np.int), plot_dat[:, 2], color='r', alpha=alpha, s=pt_size, label="Non-permanent Accuracy")
    axs[i].scatter(plot_dat[:, 6].astype(np.int), plot_dat[:, 3], color='k', s=pt_size, label="Adjusted Accuracy")
    axs[i].scatter(plot_dat[:, 6].astype(np.int), plot_dat[:, 0], color='k', alpha=alpha, s=pt_size, label="Overall Accuracy")
    axs[i].set_title('Willow-Whitehorse ' + str(i + 1).zfill(2), fontsize=10)
    axs[i].set_ylabel('Accuracy Value')
    print('accuracy values for best day thresh', np.round(plot_dat[best_threshold_days[i], :], 4))# print best accuracy values

fig.subplots_adjust(bottom=0.15, hspace=0.4, top=0.95, right=0.95, left=0.15)
plt.xlabel('Non-permanent Threshold (days)')
plt.setp(axs, xlim=(-1, 21))
plt.xticks(np.arange(0, 22, 2))
syms, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(syms, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
plt.show()


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