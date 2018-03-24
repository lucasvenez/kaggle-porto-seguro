from __future__ import print_function
from time import gmtime, strftime

import tensorflow as tf
import normalization as norm
import pandas as pd
import numpy as np
import data
import metric
import model

is_local_train = False

# Basic tf setting
print("Loading input...")

columns = ['target',
           'ps_car_13',
           'ps_reg_03',
           'ps_car_14',
           'ps_car_11_cat',
           'ps_car_12',
           'ps_car_01_cat',
           'ps_ind_05_cat',
           'ps_reg_02',
           'ps_ind_06_bin',
           'ps_ind_07_bin',
           'ps_car_04_cat',
           'ps_ind_03',
           'ps_car_15',
           'ps_car_06_cat',
           'ps_ind_17_bin',
           'ps_car_07_cat',
           'ps_car_03_cat',
           'ps_car_02_cat',
           'ps_reg_01',
           'ps_ind_16_bin',
           'ps_car_09_cat',
           'ps_ind_15',
           'ps_car_05_cat',
           'ps_car_11',
           'ps_car_08_cat',
           'ps_ind_01',
           'ps_ind_04_cat',
           'distance',
           #'angle',
           #'ft_ps_id',
           #'ft_elm',
           #'ft_xgb',
           #'ft_pca',
           #'ft_x',
           #'ft_y',
           #'ft_xy',
           'kinetic_1',
           'kinetic_2',
           'kinetic_3',
           'kinetic_4',
           'ft_x1',
           'ft_x2',
           'ft_x3'
           ]

#columns_dae = ['ft_dae_' + ('0' + str(i) if i < 10 else str(i)) for i in range(1, 81)]
#columns += columns_dae

# Get input

train_x, train_y, test_x, test_y, train_id, test_id = None, None, None, None, None, None

x_test, y_test = None, None

if is_local_train:
    train_x, train_y, train_id = data.load_train_data_with_sample(columns)
    _, _, test_x, test_y, _, test_id = data.load_train_test_sample(columns)
    x_test, y_test = test_x, test_y
else:
    train_x, train_y, train_id = data.load_train_data_with_sample(columns)
    test_x, test_id = data.load_test_data(columns[1:])
    _, _, x_test, y_test, _, _ = data.load_train_test_sample(columns, .99)

print(train_x.shape)

#
# Processing input
#
print('Normalizing...')

normalization = norm.MinMax()
train_x = normalization.fit_and_normalize(train_x)
test_x = normalization.normalize(test_x)
x_test = normalization.normalize(x_test)

#
# Building the model
#
print("Training...")

mlp = model.MLP(len(columns)-1, [30], 1, tf.nn.relu)
mlp.optimize(train_x, train_y, steps=10000, x_test=x_test, y_test=y_test)

#
# Testing model
#
print('Predicting...')
test_y_hat = mlp.predict(test_x)

if not is_local_train:
    print("Exporting...")
    result = test_y_hat
    pd.DataFrame({'id': test_id, 'target': result}).to_csv('output/prediction/' + strftime("%Y%m%d%H%M%S", gmtime()) + '-ddn-porto-seguro-train.csv', index=False)

else:
    gini = metric.Gini()
    print(gini.calculate(test_y.T, np.array(test_y_hat).T))