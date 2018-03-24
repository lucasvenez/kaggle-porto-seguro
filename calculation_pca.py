import numpy as np
import pandas as pd
import data
from sklearn.decomposition import PCA

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
           ]

print('Loading data...')

train_x, train_y, train_id = data.load_train_data(columns)
test_x, test_id = data.load_test_data(columns[1:])

print('Calculating PCA...')
pca = PCA(n_components=3)

train_x = pca.fit_transform(train_x)
train_x = np.array(train_x)

test_x = pca.transform(test_x)
test_x = np.array(test_x)

print('Exporting feature...')

df = pd.DataFrame({'ft_pca_var_01': train_x[:,0], 'ft_pca_var_02': train_x[:,1], 'ft_pca_var_03': train_x[:,2], 'id': train_id})
df.to_csv('output/ft_pca_vars_train.csv', sep=',', index=False)

df = pd.DataFrame({'ft_pca_var_01': test_x[:,0], 'ft_pca_var_02': test_x[:,1], 'ft_pca_var_03': test_x[:,2], 'id': test_id})
df.to_csv('output/ft_pca_vars_test.csv', sep=',', index=False)