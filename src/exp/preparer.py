
import glob
import os
import sys

import yaml
import pandas as pd


sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

with open('src/exp/config.yml', 'r') as fd:
    exp_params = yaml.safe_load(fd)

RAW_DATA_PATH = exp_params['paths']['data']['raw']
PREPARED_DATA_PATH = exp_params['paths']['data']['prepared']
XY_PREPARED_DATA_PATH = exp_params['paths']['data']['xy_prepared']


print('Preparation stage')
print('    Loading raw data')
files = glob.glob(os.path.join(RAW_DATA_PATH, '*.wav'))


print('    Preparing dataset')
xy =[]
for file in files:
    file_name = file.split('/')[-1]
    interpretation_type = file_name.split('_')[1]
    xy.append([file_name,interpretation_type == 'hum'])    


if not os.path.exists(PREPARED_DATA_PATH):
    os.makedirs(PREPARED_DATA_PATH)

pd.DataFrame(xy).to_csv(XY_PREPARED_DATA_PATH, index=False, header=False)
print('    Prepared dataset ready')