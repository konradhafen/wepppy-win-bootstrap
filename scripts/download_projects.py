from download_weppcloud_project import *
import pandas as pd


out_dir = 'E:/konrad/Projects/usgs/hjandrews/wepp/'
input_fn = '../project-names.csv'
df = pd.read_csv(input_fn)

for i in range(df.shape[0]):
    download_weppcloud_project(df.iloc[i, 2], out_dir)
    os.rename(out_dir + df.iloc[i, 2], out_dir + df.iloc[i, 3])