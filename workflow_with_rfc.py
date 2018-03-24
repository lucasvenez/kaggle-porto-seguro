from __future__ import print_function
from time import gmtime, strftime

import sklearn.ensemble as ensemble
import normalization as norm
import pandas as pd
import numpy as np
import data
import metric
import matplotlib.pyplot as plt

is_local_train = True
with_plot = False

# Basic tf setting
print("Loading input...")

columns = ['target',
           #'id',
           #'ps_car_13',
           #'ps_reg_03',
           #'ps_car_14',
           #'ps_car_11_cat',
           #'ps_car_12',
           #'ps_car_01_cat',
           #'ps_ind_05_cat',
           #'ps_reg_02',
           #'ps_ind_06_bin',
           #'ps_ind_07_bin',
           #'ps_car_04_cat',
           #'ps_ind_03',
           #'ps_car_15',
           #'ps_car_06_cat',
           #'ps_ind_17_bin',
           #'ps_car_07_cat',
           #'ps_car_03_cat',
           #'ps_car_02_cat',
           #'ps_reg_01',
           #'ps_ind_16_bin',
           #'ps_car_09_cat',
           #'ps_ind_15',
           #'ps_car_05_cat',
           #'ps_car_11',
           #'ps_car_08_cat',
           #'ps_ind_01',
           #'ps_ind_04_cat',
           'distance',
           #'angle',
           #ft_ps_id',
           'ft_elm',
           'ft_mlp'
           #'ft_pca',
           #'ft_ddn',
           #'ft_x',
           #'ft_y',
           #'ft_xy'
           ]

#columns_dae = ['ft_dae_' + ('0' + str(i) if i < 10 else str(i)) for i in range(1, 81)]

#columns += columns_dae

# Get input

train_x, train_y, test_x, test_y, train_id, test_id = None, None, None, None, None, None

if is_local_train:
    train_x, train_y, test_x, test_y, train_id, test_id = data.load_train_test_sample(columns)
else:
    train_x, train_y, train_id = data.load_train_data(columns)
    test_x, test_id = data.load_test_data(columns[1:])

#
# Processing input
#
print('Normalizing...')

normalization = norm.MinMax()
#train_x = normalization.fit_and_normalize(train_x)#, train_y[:,0])
#test_x  = normalization.normalize(test_x)

#
# Building the model
#
print("Training...")

rfc = ensemble.RandomForestRegressor(100000, n_jobs=8, random_state=9)

rfc.fit(train_x, train_y)

#
# Testing model
#
print("Predicting...")
test_y_hat = rfc.predict(test_x)

print('Min: ', min(test_y_hat))
print('Max: ', max(test_y_hat))

test_y_hat = [1. if y_hat > 1. else y_hat for y_hat in test_y_hat]

if not is_local_train:
    #print("Exporting test...")
    #result = test_y_hat
    #pd.DataFrame({'id': test_id, 'ft_xgb': result}).to_csv('output/ft_xgb_test.csv', index=False)

    print("Exporting...")
    result = test_y_hat
    df = pd.DataFrame({'id': test_id, 'target': result})
    df.to_csv('output/prediction/' + strftime("%Y%m%d%H%M%S", gmtime()) + '-xgb-porto-seguro.csv', index=False, sep=',')

else:
    gini = metric.Gini()
    print(gini.calculate(test_y.T, np.array([test_y_hat])))