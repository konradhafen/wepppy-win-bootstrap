import numpy as np
import spotpy
import os
import matplotlib.pyplot as plt

wd = "E:/konrad/Projects/usgs/hjandrews/wepp/hja-ws1-base/"
output = os.path.join(wd, 'export/calibration_results_annual_sens6p.npy')
nvars = 6
nrow = 3
ncol = 2
xlabs = ['Crop coefficient', 'K (restrictive layer)', 'Deep seepage coefficient', 'Baseflow coefficient', 'Field capacity', 'Percent rock']
xrange = [(0.8, 1.2), (0.005, 1000), (0.0, 0.2), (0.0, 0.2), (0.0, 0.8), (0, 80)]

data = np.asarray(np.load(output).tolist())
dat_plt = data[np.where((data[:, 0] > -15.0) & (data[:, 0] < 15.0))]

f, ax = plt.subplots(nrow, ncol, sharex=False)

for i in range(nrow):
    for j in range(ncol):
        index = i*ncol+j
        ax[i, j].hist(dat_plt[:, index+1], range=xrange[index], bins=40)
        ax[i, j].set_xlabel(xlabs[index])
plt.subplots_adjust(hspace=0.3)
plt.show()