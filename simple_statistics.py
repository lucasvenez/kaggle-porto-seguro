from __future__ import print_function
import math
import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

columns = ['ps_car_13',
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
           'angle',
           'ft_dae_01',
           'ft_dae_02',
           'ft_dae_03',
           'ft_dae_04',
           'ft_dae_05',
           'ft_dae_06',
           'ft_dae_07',
           'ft_dae_08',
           'ft_dae_09',
           'ft_dae_10',
           'ft_dae_11',
           'ft_dae_12',
           'ft_dae_15',
           'ft_ps_id',
           'ft_elm',
           'ft_pca'
           ]

print('Loading data...')
train = pd.read_csv('./input/test.csv', sep=',')

features_dis = pd.read_csv('./input/ft_distance_vars_test.csv', sep=',')
features_dae = pd.read_csv('./output/ft_dae_test.csv', sep=',')
features_psi = pd.read_csv('./output/ft_ps_id_test.csv', sep=',')
features_pca = pd.read_csv('./output/ft_pca_test.csv', sep=',')
features_elm = pd.read_csv('./output/ft_elm_test.csv', sep=',')

train = train.merge(features_dis, on='id', how='inner')
train = train.merge(features_dae, on='id', how='inner')
train = train.merge(features_psi, on='id', how='inner')
train = train.merge(features_pca, on='id', how='inner')
train = train.merge(features_elm, on='id', how='inner')

train_id = train['id']

train = train[columns]

def normalize(serie):
    return (serie - min(serie)) / (max(serie) - min(serie))

train = train.apply(normalize)

for column in train.columns:
    if column != 'target':
        train[column] = train[column] + abs(min(train[column]))

print('Calculating radviz...')

r = pd.DataFrame()
id = []
__x  = []
__y  = []
__xy = []

m = train.shape[1]

s = np.array([(np.cos(t), np.sin(t))
                  for t in [2.0 * np.pi * (i / float(m))
                            for i in range(m)]])

n = len(train)

for i in range(n):
    row = train.iloc[i].values
    row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
    y = (s * row_).sum(axis=0) / row.sum()
    __x.append(y[0])
    __y.append(y[1])
    __xy.append(math.sqrt(y[0]*y[0] + y[1]*y[1]))

print('Exporting...')

pd.DataFrame({'id': train_id, 'ft_x': __x, 'ft_y': __y, 'ft_xy': __xy}).to_csv('output/ft_xy_test.csv', sep=',', index=False)
