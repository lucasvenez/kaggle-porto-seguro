from __future__ import print_function
from time import gmtime, strftime

import tensorflow as tf
import normalization as norm
import pandas as pd
import numpy as np
import data
import metric

is_local_train = False
with_plot = False

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
           #ft_ps_id',
           'ft_elm',
           #'ft_xgb',
           #'ft_pca',
           #'ft_x',
           #'ft_y',
           #'ft_xy'
           ]

#columns_dae = ['ft_dae_' + ('0' + str(i) if i < 10 else str(i)) for i in range(1, 81)]
#columns += columns_dae

# Get input

train_x, train_y, test_x, test_y, train_id, test_id = None, None, None, None, None, None

if is_local_train:
    train_x, train_y, train_id = data.load_train_data_with_sample(columns)
    test_x, test_y, _, _, test_id, _ = data.load_train_test_sample(columns)

else:
    train_x, train_y, train_id = data.load_train_data_with_sample(columns)
    test_x, test_id = data.load_test_data(columns[1:])

print(train_x.shape)
print(test_x.shape)

#
# Processing input
#
print('Normalizing...')

with tf.device('/cpu:0'):
    normalization = norm.MinMax()
    normalization.fit(train_x)
    train_x, train_y, train_id = data.load_train_data(columns)
    train_x = normalization.normalize(train_x)
    test_x = normalization.normalize(test_x)

#
# Building the model
#
print('Loading trained model...')

session = tf.Session()

id = '20171127222814-mlp-porto-seguro-221-30500'

with tf.device('/cpu:0'):

    saver = tf.train.import_meta_graph('./output/model/{}.meta'.format(id), clear_devices=True)
    saver.restore(session, "./output/model/{}".format(id))

    input = tf.get_default_graph().get_tensor_by_name('input:0')
    keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

    model = tf.get_default_graph().get_tensor_by_name('Add_3:0')

    #
    # Testing model
    #
    print("Predicting...")
    train_y_hat = session.run(model, feed_dict={input: train_x, keep_prob: 1.})
    test_y_hat = session.run(model, feed_dict={input: test_x, keep_prob: 1.})
    print(test_y_hat.shape)

    print('Min: ', min(test_y_hat))
    print('Max: ', max(test_y_hat))

train_y_hat = [1. if y_hat > 1. else (.0 if y_hat < 0 else y_hat) for y_hat in train_y_hat[:,0]]
test_y_hat = [1. if y_hat > 1. else (.0 if y_hat < 0 else y_hat) for y_hat in test_y_hat[:,0]]

if not is_local_train:

    print("Exporting train...")
    pd.DataFrame({'id': train_id, 'ft_mlp': train_y_hat}).to_csv('output/ft_mlp_train.csv', index=False)

    print("Exporting test...")
    pd.DataFrame({'id': test_id, 'ft_mlp': test_y_hat}).to_csv('output/ft_mlp_test.csv', index=False)

else:
    gini = metric.Gini()
    print(gini.calculate(test_y.T, np.array([test_y_hat])))