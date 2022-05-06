import glob
import os
import sys

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from pred.transformation import transformation


with open('src/exp/config.yml', 'r') as fd:
    exp_params = yaml.safe_load(fd)

PREPARED_DATA_PATH = exp_params['paths']['data']['prepared']
XY_PREPARED_DATA_PATH = exp_params['paths']['data']['xy_prepared']
TRANSFORMED_DATA_PATH = exp_params['paths']['data']['transformed']
X_TRANSFORMED_DATA_PATH = exp_params['paths']['data']['x_transformed']
Y_TRANSFORMED_DATA_PATH = exp_params['paths']['data']['y_transformed']

print('Transformation stage')
print('    Extracting features')
xy = pd.read_csv(XY_PREPARED_DATA_PATH, index_col = 0, header = None)
X,y =[],[]
for fileID in tqdm(xy.index):
    yi = xy.loc[fileID]
    xi = transformation(fileID)

    X.append(xi)
    y.append(yi)


if not os.path.exists(TRANSFORMED_DATA_PATH):
    os.makedirs(TRANSFORMED_DATA_PATH)

np.savetxt(X_TRANSFORMED_DATA_PATH, np.array(X), delimiter=",")
np.savetxt(Y_TRANSFORMED_DATA_PATH, np.array(y), delimiter=",")

print('    Transformed dataset ready')