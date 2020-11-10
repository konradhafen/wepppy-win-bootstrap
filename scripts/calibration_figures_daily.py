import numpy as np
import matplotlib.pyplot as plt
import os


proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
results_fn = "export/calibration_results_daily.npy"
eval_fn = "export/calibration_results_daily_eval.npy"

wsn = 2  # watershed number
nyears = 2  # number of years to show

proj_dir = "hja-ws" + str(wsn) + "-base"
simulation = np.asarray(np.load(os.path.join(proj_base, proj_dir, results_fn)).tolist())
print(simulation.shape)
print(simulation[:8])
evaluation = np.asarray(np.load(os.path.join(proj_base, proj_dir, eval_fn)))
print(evaluation.shape)
# print(simulation[6:].shape, simulation[6:10])
# print(evaluation.shape, evaluation[:4])
plt.plot(evaluation[-365*nyears:], label='evaluation')
plt.plot(simulation[-365*nyears:], label='simulation')
# results = results[results[:, 1].argsort()]
# plt.plot(results[:, 1], results[:, 0], label="WS"+str(i))
plt.ylabel('cfs')
plt.xlabel('Days')
plt.legend()
plt.show()
