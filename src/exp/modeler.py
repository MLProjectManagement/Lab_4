import os
import sys
import pickle

import numpy as np
import yaml

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from pred.model import model


with open('src/exp/config.yml', 'r') as fd:
    exp_params = yaml.safe_load(fd)


X_TRAIN_PATH = exp_params['paths']['data']['x_train']
Y_TRAIN_PATH = exp_params['paths']['data']['y_train']

MODEL_PATH = exp_params['paths']['model']['root']
TRAINED_MODEL = exp_params['paths']['model']['trained_model']


print('Modelling stage')
print('    Loading training data')
X_train = np.loadtxt(X_TRAIN_PATH, delimiter = ",")
y_train = np.loadtxt(Y_TRAIN_PATH, delimiter = ",")

print('    Training model')
model.fit(X_train, y_train)


print('    Saving model')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

with open(TRAINED_MODEL, "wb") as fd:
    pickle.dump(model, fd)
