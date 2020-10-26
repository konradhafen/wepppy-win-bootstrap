import numpy as np
import matplotlib.pyplot as plt
import os


proj_dir = "E:/konrad/Projects/usgs/hjandrews/wepp/hja-ws1-base/"
results_fn = "export/calibration_results_annual.npy"
fn = os.path.join(proj_dir, results_fn)
results = np.asarray(np.load(fn).tolist())
results = results[results[:, 1].argsort()]

plt.plot(results[:, 1], results[:, 0])
plt.ylabel('PBIAS (water year yield)')
plt.xlabel('Crop coefficient')
plt.show()
