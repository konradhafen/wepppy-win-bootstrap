import numpy as np
import matplotlib.pyplot as plt
import os


proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
results_fn = "export/calibration_results_daily.npy"
eval_fn = "export/calibration_results_daily_eval.npy"

wsn = 2  # watershed number
nyears = 2  # number of years to show
n_pre_col = 7 # number of columns before simulation values begin
irow = 0 # index of row to plot

proj_dir = "hja-ws" + str(wsn) + "-base"
simulation = np.asarray(np.load(os.path.join(proj_base, proj_dir, results_fn)).tolist())

# Plot objective function values (pbias, nash-sutcliffe, nash-sutcliffe on log of Q)
objs = simulation[:, :7]
s = objs[objs[:, 0].argsort()]
i = np.where(np.fabs(objs[:, 0]) < 20.0)
s = objs[i[0], :]
# parkc,parkr,parks,parkb
plt.plot(s[:, 3], label='kc')
# plt.plot(s[:, 4], label='kr')
plt.plot(s[:, 5], label='ks')
plt.plot(s[:, 6], label='kb')
plt.legend()
plt.show()
print(s.shape)
s = s[s[:, 0].argsort()]
plt.plot(np.fabs(s[:, 0]/100.0), label='PBIAS')
plt.plot(s[:, 1], label='NSE')
plt.plot(s[:, 2], label='NSE (log Q)')
plt.ylabel('Value')
plt.xlabel('Index')
plt.legend()
plt.show()


# Plots simulated vs observed hydrographs
evaluation = np.asarray(np.load(os.path.join(proj_base, proj_dir, eval_fn)))
plt.plot(evaluation[-365*nyears:], label='evaluation')
plt.plot(simulation[irow, -365*nyears:], label='simulation')
# results = results[results[:, 1].argsort()]
# plt.plot(results[:, 1], results[:, 0], label="WS"+str(i))
plt.ylabel('cfs')
plt.xlabel('Days')
plt.legend()
plt.show()
