import numpy as np
import os
import matplotlib.pyplot as plt


proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
results_fn = "export/calibration_results_annual_perm.npy"
eval_fn = "export/calibration_results_annual_perm_eval.npy"

proj_names = ['ww-ws1-base', 'ww-ws2-base', 'ww-ws3-base', 'ww-ws4-base']

fig, axs = plt.subplots(len(proj_names), sharex=True, sharey=True, figsize=(5, 7))
n_pre_col = 5
alpha = 0.5
pt_size = 20.0

for i in range(len(proj_names)):
    simulation = np.asarray(np.load(os.path.join(proj_base, proj_names[i], results_fn)).tolist())
    evaluation = np.asarray(np.load(os.path.join(proj_base, proj_names[i], eval_fn)).tolist())
    accuracy = np.zeros((simulation.shape[0], 3))
    accuracy[:, 0] = simulation[:, 0]
    eval_dry = np.where(evaluation == 0)
    eval_wet = np.where(evaluation == 1)
    sims = simulation[:, n_pre_col:-1]
    sims_dry = sims[:, eval_dry[0]]
    accuracy[:, 1] = 1.0 - sims_dry.sum(axis=1)/len(eval_dry[0])
    sims_wet = sims[:, eval_wet[0]]
    accuracy[:, 2] = sims_wet.sum(axis=1)/len(eval_wet[0])
    print(proj_names[i])
    print(np.unique(accuracy, axis=0))
    plot_dat = np.unique(accuracy, axis=0)
    axs[i].scatter(plot_dat[:, 0], plot_dat[:, 1], color='r', alpha=alpha, s=pt_size, label="Non-permanent Accuracy")
    axs[i].scatter(plot_dat[:, 0], plot_dat[:, 2], color='b', alpha=alpha, s=pt_size, label="Permanent Accuracy")
    axs[i].scatter(plot_dat[:, 0], plot_dat[:, 0] - np.fabs(plot_dat[:, 1] - plot_dat[:, 2]), color='k', s=pt_size, label="Adjusted Accuracy")
    axs[i].set_title('Willow-Whitehorse ' + str(i+1).zfill(2) + ' (n=' + str(sims.shape[1]) + ")", fontsize=10)

fig.subplots_adjust(bottom=0.15, hspace=0.4, top=0.95, right=0.95, left=0.1)
plt.xlabel('Overall Accuracy')
plt.xlim((0.0, 1.0))
plt.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -1.0))
plt.show()