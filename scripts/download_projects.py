from download_weppcloud_project import *
import pandas as pd
import shutil


out_dir = 'E:/konrad/Projects/usgs/hjandrews/wepp/'
input_fn = '../project-names.csv'
df = pd.read_csv(input_fn)
for i in range(1):
# for i in range(df.shape[0]):
    download_weppcloud_project(df.iloc[i, 2], out_dir)
    if os.path.isdir(out_dir + df.iloc[i, 3]):
        shutil.rmtree(out_dir + df.iloc[i, 3])
    os.rename(out_dir + df.iloc[i, 2], out_dir + df.iloc[i, 3])
    print(i + 1, 'completed')