import numpy as np
import matplotlib.pyplot as plt
import os


proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
results_fn = "export/calibration_results_annual.npy"
entries = []

for i in range(1, 4):
    proj_dir = "hja-ws" + str(i) + "-base"
    fn = os.path.join(proj_base, proj_dir, results_fn)
    results = np.asarray(np.load(fn).tolist())
    results = results[results[:, 1].argsort()]
    plt.plot(results[:, 1], results[:, 0], label="WS"+str(i))
plt.ylabel('PBIAS (water year yield)')
plt.xlabel('Crop coefficient')
plt.legend()
plt.show()
