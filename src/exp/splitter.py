import sys
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )


with open('src/exp/config.yml', 'r') as fd:
    exp_params = yaml.safe_load(fd)

TRANSFORMED_DATA_PATH = exp_params['paths']['data']['transformed']
X_TRANSFORMED_DATA_PATH = exp_params['paths']['data']['x_transformed']
Y_TRANSFORMED_DATA_PATH = exp_params['paths']['data']['y_transformed']

TRAIN_DATA_PATH = exp_params['paths']['data']['train']
TEST_DATA_PATH = exp_params['paths']['data']['test']
X_TRAIN_PATH = exp_params['paths']['data']['x_train']
X_TEST_PATH = exp_params['paths']['data']['x_test']
Y_TRAIN_PATH = exp_params['paths']['data']['y_train']
Y_TEST_PATH = exp_params['paths']['data']['y_test']

TEST_SIZE = exp_params['splitter']['test_size']
RANDOM_STATE = exp_params['splitter']['split_seed']

print('Splitting stage')

X = np.loadtxt(X_TRANSFORMED_DATA_PATH, delimiter = ",")
y = np.loadtxt(Y_TRANSFORMED_DATA_PATH, delimiter = ",")

print('    Splitting transformed dataset')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

if not os.path.exists(TRAIN_DATA_PATH):
    os.makedirs(TRAIN_DATA_PATH)

if not os.path.exists(TEST_DATA_PATH):
    os.makedirs(TEST_DATA_PATH)

print('    Saving training and test datasets')
np.savetxt(X_TRAIN_PATH, X_train, delimiter=",")
np.savetxt(X_TEST_PATH, X_test, delimiter=",")
np.savetxt(Y_TRAIN_PATH, y_train, delimiter=",")
np.savetxt(Y_TEST_PATH, y_test, delimiter=",")

print('    Training and test datasets ready')