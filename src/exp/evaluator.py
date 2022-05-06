import sys
import os
import pickle
import time

from sklearn.metrics import accuracy_score
import numpy as np
import dagshub
import yaml

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

with open('src/exp/config.yml', 'r') as fd:
    exp_params = yaml.safe_load(fd)

X_TEST_PATH = exp_params['paths']['data']['x_test']
Y_TEST_PATH = exp_params['paths']['data']['y_test']

TRAINED_MODEL = exp_params['paths']['model']['trained_model']

LOGS_PATH = exp_params['paths']['logs']


print('Evaluation stage')
print('    Loading trained model')

with open(TRAINED_MODEL, "rb") as fd:
    model = pickle.load(fd)

print('    Loading test data')
X_test = np.loadtxt(X_TEST_PATH, delimiter = ",")
y_test = np.loadtxt(Y_TEST_PATH, delimiter = ",")



print('    Obtaining performance report')
with dagshub.dagshub_logger(metrics_path="logs/test_metrics.csv", hparams_path="logs/params.yml") as logger: 

    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)

    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    logger.log_metrics({f'accuracy':score})

    logger.log_hyperparams(model_class=type(model).__name__)
    logger.log_hyperparams({'model': model.get_params()})

print('    Performance report ready')