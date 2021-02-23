import os
import numpy as np
from run_project import *
from sol_prep import *
from snow_prep import *

proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/"
in_csv = os.path.join(proj_base, "hja_all_calibration/gof_params.csv")
proj_names = ["hja-ws1-base", "hja-ws2-base", "hja-ws3-base", "hja-ws6-base", "hja-ws7-base", "hja-ws8-base", "hja-ws9-base", "hja-ws10-base", "hja-mack-base"]

params = np.loadtxt(in_csv, delimiter=",")
for i in range(len(proj_names)):
    print('params, kc:', params[i, 3], 'kb:', params[i, 4], 'ks:', params[i, 5], 'kr:', params[i, 6])
    pmet_coeffs = [params[i, 3], 0.8]
    gw_coeffs = [200.0, params[i, 4], params[i, 5], 1.0001]
    snow_coeffs = [-2.0, 100.0, 250.0]
    soil_prep(os.path.join(proj_base, proj_names[i]), kr=params[i, 6], field_cap=None, pct_rock=None)
    result = run_project(os.path.join(proj_base, proj_names[i]), numcpu=8, gwcoeff=gw_coeffs, pmet=pmet_coeffs, snow=snow_coeffs)