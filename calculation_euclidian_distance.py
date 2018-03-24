from __future__ import print_function
from features import MeanEuclidianDistance
import pandas as pd
import numpy as np

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
           'ps_ind_04_cat'
          ]

df_sample_source = pd.read_csv('./input/train.csv', sep=',')
fraud_sample = df_sample_source[df_sample_source['target'] == 1][columns[1:]]

df = pd.read_csv('./input/test.csv', sep=',')
ids = df['id']

del df['id']

df = df[columns[1:]].as_matrix()

result = np.empty([0, 3], dtype=np.float64)

i = 1

step = 100

med = MeanEuclidianDistance()

for index in range(0, len(ids), step):

   sample = np.array([[row] for row in df[index:(min(len(ids), index + step)),:]])

   current_result = np.array(med.calculate(sample, fraud_sample)).T

   end = (min(len(ids), index + step))

   current_result = np.insert(current_result, 0, ids[index:end], axis=1)

   if i % 100 == 0:
      print(end, ' of ', len(ids))

   result = np.append(result, current_result, axis=0)

   i += 1

features = pd.DataFrame(result, columns=['id', 'distance', 'angle'])

features.to_csv('output/ft_distance_vars_test.csv', index=False, sep=',')
