import numpy as np
import matplotlib.pyplot as plt
import os
from textwrap import wrap
import pandas as pd
from datetime import datetime


proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
results_fn = "export/calibration_results_daily.npy"
eval_fn = "export/calibration_results_daily_eval.npy"

pbias_thresh = 25.0
lnse_thresh = 0.3
nse_thresh = 0.3
nyears = 2  # number of years to show
n_pre_col = 7  # number of columns before simulation values begin
cfs_to_Ls = 28.3168

# create dates for plotting
end_date = datetime.strptime("2019-09-30", "%Y-%m-%d")
date_list = pd.date_range(end=end_date, periods=nyears * 365)
# end_date_index = end_date.year * 1000 + end_date.timetuple().tm_yday
# date_index = end_date - (np.arange(-365 * 2 + 1, 1) * -1) * np.timedelta64(1, 'D')

proj_names = ["hja-ws1-base", "hja-ws2-base", "hja-ws3-base", "hja-ws6-base", "hja-ws7-base", "hja-ws8-base", "hja-ws9-base", "hja-ws10-base", "hja-mack-base"]
ws_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ws_id = [1, 2, 3, 6, 7, 8, 9, 10, "MACK"]

proj_names = ["hja-ws1-base", "hja-ws2-base", "hja-ws3-base", "hja-ws6-base", "hja-ws7-base", "hja-ws8-base", "hja-ws9-base", "hja-ws10-base"]
ws_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ws_id = [1, 2, 3, 6, 7, 8, 9, 10]

par_kc = []
par_ks = []
par_kr = []
par_kb = []
acc_pbias = []
acc_nse = []
acc_lnse = []
x_par = []
x_pbias = []
x_nse = []
x_lnse = []

i_ws = 0 # index of row to plot
best_sims = None
evals = None

for proj_name in proj_names:
    simulation = np.asarray(np.load(os.path.join(proj_base, proj_name, results_fn)).tolist())
    evaluation = np.asarray(np.load(os.path.join(proj_base, proj_name, eval_fn)).tolist())
    if best_sims is None:
        best_sims = np.full((len(proj_names), simulation.shape[1]), np.nan)
    if evals is None:
        evals = np.empty((len(proj_names), evaluation.shape[0]))
    evals[i_ws, :] = evaluation

    objs = simulation[:, :n_pre_col]
    s = objs[objs[:, 0].argsort()]
    minval = np.min(np.fabs(objs[:, 0]))
    # print(proj_name, 'pbias min', objs[np.where(np.fabs(objs[:, 0]) == np.min(np.fabs(objs[:, 0]))), 0], 'nse max', np.max(objs[:, 1]), 'log nse max', np.max(objs[:, 2]))
    i = np.where((np.fabs(objs[:, 0]) < pbias_thresh) & (objs[:, 1] > nse_thresh) & (objs[:, 2] > lnse_thresh))
    if len(i[0]) == 0:  # if not all parameter constraints are met, get parameters corresponding to max value of NSE log Q
        i = np.where((objs[:, 2] == np.max(objs[:, 2])))

    s = objs[i[0], :]
    sim_obj = simulation[i[0], :]
    if sim_obj.shape[0] > 0:
        i_best = np.where(sim_obj[:, 2] == np.max(sim_obj[:, 2]))
        # print(proj_name, "ibest shape, value", i_best[0].shape, i_best)
        # print(simulation.shape, sim_obj[i_best[0], :].shape, best_sims[i_ws, :].shape)
        best_sims[i_ws, :] = sim_obj[i_best[0], :]
        # print(evals[i_ws, :].shape, sim_obj[i_best[0], n_pre_col+1:][0].shape)
        # print('pbias', spotpy.objectivefunctions.pbias(evals[i_ws, :], sim_obj[i_best[0], n_pre_col+1:][0]), sim_obj[i_best[0], 0])
    n = s[:, 3].shape[0]
    par_kc.extend(s[:, 3])
    par_kr.extend(s[:, 4])
    par_ks.extend(s[:, 5])
    par_kb.extend(s[:, 6])
    x_par.extend(np.repeat(ws_nums[i_ws], s[:, 3].shape[0]))
    acc_pbias.extend(s[:, 0]/100.0)
    x_pbias.extend(np.repeat(ws_nums[i_ws] - 0.2, s[:, 0].shape[0]))
    acc_nse.extend(s[:, 1])
    x_nse.extend(np.repeat(ws_nums[i_ws], s[:, 0].shape[0]))
    acc_lnse.extend(s[:, 2])
    x_lnse.extend(np.repeat(ws_nums[i_ws] + 0.2, s[:, 0].shape[0]))
    i_ws += 1

# goodness of fit plot
alpha = 0.4
plt.scatter(x_pbias, acc_pbias, alpha=alpha, label="PBIAS")
plt.scatter(x_nse, acc_nse, alpha=alpha, label="NSE")
plt.scatter(x_lnse, acc_lnse, alpha=alpha, label="NSE log(Q)")
plt.vlines([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], -0.5, 1.0, linewidth=0.5)
plt.ylim(top=0.75, bottom=-0.25)
plt.xlabel("Watershed ID")
plt.ylabel("Value")
plt.xticks(ticks=ws_nums, labels=ws_id)
plt.legend()
plt.show()

# parameter values plot
alpha = 0.3
fig, axs = plt.subplots(4, 1, figsize=(5, 7), sharex=True)
axs[0].scatter(x_par, par_kc, alpha=alpha, color='k')
axs[0].set_ylabel("\n".join(wrap("Crop coefficient", 15)))
axs[1].scatter(x_par, par_kb, alpha=alpha, color='k')
axs[1].set_ylabel("\n".join(wrap("Baseflow coefficient ($d^{-1}$)", 15)))
axs[2].scatter(x_par, par_ks, alpha=alpha, color='k')
axs[2].set_ylabel("\n".join(wrap("Deep seepage coefficient ($d^{-1}$)", 15)))
axs[3].scatter(x_par, par_kr, alpha=alpha, color='k')
axs[3].set_ylabel("\n".join(wrap("Restrictive layer conductivity ($mm$ $hr^{-1}$)", 17)))
axs[3].set_xlabel("Watershed ID")
axs[3].set_xticks(ws_nums)
axs[3].set_xticklabels(ws_id)
plt.subplots_adjust(left=0.22, right=0.95, hspace=0.1, bottom=0.08, top=0.95)
plt.show()

# hydrograph comparison plot
nrow = 4
ncol = 2
lw = 1.0
fig, axs = plt.subplots(nrow, ncol, figsize=(9, 7), sharex=True)
for i in range(best_sims.shape[0]):
    row = i // ncol
    col = i % ncol
    axs[row, col].plot(date_list[:-1], evals[i, -365 * nyears:-1] * cfs_to_Ls, linewidth=lw)
    axs[row, col].plot(date_list[:-1], best_sims[i, -365 * nyears:-1] * cfs_to_Ls, linewidth=lw, ls="--")
    axs[row, col].set_title("WS" + str(ws_id[i]).zfill(2), loc="left")
    axs[row, col].annotate("PB: " + str(best_sims[i, 0]) + "\nNSE: " + str(best_sims[i, 1]) + "\nNSE (log Q): " + str(best_sims[i, 2]), xy=(0.02, 0.95), xycoords='axes fraction', verticalalignment='top')
    if row == (nrow - 1):
        # axs[row, col].set_xlabel("Date")
        axs[row, col].tick_params('x', labelrotation=45)
    if col == 0:
        axs[row, col].set_ylabel("L/s")
plt.subplots_adjust(hspace=0.3, wspace=0.13, bottom=0.12, top=0.95, right=0.95, left=0.1)
plt.show()

df = pd.DataFrame(data=best_sims[:, :n_pre_col])
df.columns = ['PBIAS', 'NSE', 'logNSE', 'CropCoef', 'KvRestrict', 'DeepSeep', 'BFRecess']
df.to_csv(os.path.join(proj_base, 'hja_all_calibration/gof_params_table.csv'), index=False)
np.savetxt(os.path.join(proj_base, 'hja_all_calibration/gof_params.csv'), best_sims[:, :n_pre_col], delimiter=",")