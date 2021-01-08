import numpy as np
import matplotlib.pyplot as plt
import os


proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
results_fn = "export/calibration_results_daily.npy"
eval_fn = "export/calibration_results_daily_eval.npy"

wsn = 1  # watershed number
nyears = 2  # number of years to show
n_pre_col = 7 # number of columns before simulation values begin

proj_dir = "hja-ws" + str(wsn) + "-base2"
proj_names = ["hja-ws1-base2", "hja-ws2-base", "hja-ws3-base", "hja-ws6-base", "hja-ws7-base", "hja-ws8-base", "hja-ws9-base", "hja-ws10-base"]
ws_nums = [1, 2, 3, 4, 5, 6, 7, 8]
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

for proj_name in proj_names:
    simulation = np.asarray(np.load(os.path.join(proj_base, proj_name, results_fn)).tolist())
    objs = simulation[:, :n_pre_col]
    s = objs[objs[:, 0].argsort()]
    i = np.where((np.fabs(objs[:, 0]) < 20.0) & (objs[:, 2] > 0.4))
    # i = np.where(objs[:, 2] > 0.4)
    s = objs[i[0], :]
    sim_obj = simulation[i[0], :]
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

alpha = 0.5
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

fig, axs = plt.subplots(4, 1, sharex=True)
axs[0].scatter(x_par, par_kc, alpha=alpha)
axs[1].scatter(x_par, par_kb, alpha=alpha)
axs[2].scatter(x_par, par_ks, alpha=alpha)
axs[3].scatter(x_par, par_kr, alpha=alpha)
axs[3].set_xticks(ws_nums)
axs[3].set_xticklabels(ws_id)

plt.show()
