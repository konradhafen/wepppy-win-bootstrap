import numpy as np
import spotpy
import os
import matplotlib.pyplot as plt

wd = "E:/konrad/Projects/usgs/hjandrews/wepp/hja-ws1-base/"
output = os.path.join(wd, 'export/calibration_results_annual_sens6p.npy')
nvars = 4
nrow = 2
ncol = 2
xlabs = ['Crop coefficient', 'K (restrictive layer)', 'Deep seepage coefficient', 'Baseflow coefficient', 'Field capacity', 'Percent rock']
xrange = [(0.85, 0.95), (0.005, 100), (0.0, 0.003), (0.0, 0.003), (0.0, 0.8), (0, 80)]

data = np.asarray(np.load(output).tolist())
# dat_plt = data[np.where((data[:, 0] > -10.0) & (data[:, 0] < 10.0) & (data[:, 1] < 1.0) & (data[:, 4] < 0.075))]
dat_plt = data[np.where((data[:, 0] > -10.0) & (data[:, 0] < 10.0))]
print(dat_plt.shape)
print(np.round(dat_plt[:, :5], 5))

f, ax = plt.subplots(nrow, ncol, sharex=False)

for i in range(nrow):
    for j in range(ncol):
        index = i*ncol+j
        ax[i, j].hist(dat_plt[:, index+1], range=xrange[index], bins=40)
        ax[i, j].set_xlabel(xlabs[index])
plt.subplots_adjust(hspace=0.3)
plt.show()