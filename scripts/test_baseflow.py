import os
import numpy as np
from run_project import *
from sol_prep import *
from snow_prep import *

proj_base = "E:/konrad/Projects/usgs/hjandrews/wepp/testing"

proj_names = ["hja-ws6-01", "hja-ws6-10", "hja-ws6-04"]
kb = [0.01, 0.1, 0.04]


for i in range(len(proj_names)):
    # print('params, kc:', params[i, 3], 'kb:', params[i, 4], 'ks:', params[i, 5], 'kr:', params[i, 6])
    # pmet_coeffs = [0.95, 0.8]
    gw_coeffs = [200.0, kb[i], 0.0, 1.0001]
    snow_coeffs = [-2.0, 100.0, 250.0]
    # soil_prep(os.path.join(proj_base, proj_names[i]), kr=params[i, 6], field_cap=None, pct_rock=None)
    result = run_project(os.path.join(proj_base, proj_names[i]), numcpu=8, gwcoeff=gw_coeffs, snow=snow_coeffs)
