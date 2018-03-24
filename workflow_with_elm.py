from __future__ import print_function
from time import gmtime, strftime

import numpy as np
import normalization as norm
import data
import pandas as pd
import xgboost as xgb
import metric
import model

is_local_train = True

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
           'ps_reg_F',
           'ps_reg_M',
           #'angle',
           #'ft_ps_id',
           #'ft_elm',
           #'ft_xgb',
           #'ft_pca',
           #'ft_x',
           #'ft_y',
           #'ft_xy',
           'ft_pca_var_01',
           'ft_pca_var_02',
           'ft_pca_var_03',
           'kinetic_1',
           'kinetic_2',
           'kinetic_3',
           'kinetic_4',
           'ft_x1',
           'ft_x2',
           'ft_x3'
           ]

#
# Loading data...
#
print("Loading data...")
train_x, train_y, test_x, test_y, train_id, test_id = None, None, None, None, None, None
x_test, y_test, id_test = None, None, None

if is_local_train:
    train_x, train_y, test_x, test_y, train_id, test_id = data.load_train_test_sample(columns)
    x_test, y_test, id_test = test_x, test_y, test_id
else:
    train_x, train_y, x_test, y_test, train_id, id_test = data.load_train_test_sample(columns, .99)
    test_x, test_id = data.load_test_data(columns[1:])

normalization= norm.MinMax()
#train_x = normalization.fit_and_normalize(train_x)
#test_x = normalization.normalize(test_x)

#
# Building the model
#
print("Training...")

df_train = pd.DataFrame({'id': train_id})
df_test  = pd.DataFrame({'id': test_id})
df_test_t= pd.DataFrame({'id': id_test})
gini = metric.Gini()

for i in range(19, 41):

    elm = model.ELM(train_x.shape[0], train_x.shape[1], i, 1)
    welm = model.WELM(train_x.shape[0], train_x.shape[1], i, 1)

    #
    # Training model
    #
    elm.feed(train_x, train_y)#, w=np.array([1., .618]))
    welm.feed(train_x, train_y, w=np.array([1., .618]))

    #
    # Testing model
    #
    y_test_hat_elm = elm.test(x_test)
    test_y_hat_elm  = elm.test(test_x)
    train_y_hat_elm = elm.test(train_x)

    y_test_hat_welm = welm.test(x_test)
    test_y_hat_welm = welm.test(test_x)
    train_y_hat_welm = welm.test(train_x)

    print('======================================================================')

    gini_elm = gini.calculate(y_test.T, y_test_hat_elm.T)
    print(i, ':', gini_elm, max(y_test_hat_elm), min(y_test_hat_elm))

    gini_welm = gini.calculate(y_test.T, y_test_hat_welm.T)
    print(i, ':', gini_welm, max(y_test_hat_welm), min(y_test_hat_welm))

    print('======================================================================')

    index = i - 9

    if gini_elm > .0 and gini_welm > .0:

        column_elm = 'elm_' + ('' if index > 99 else ('0' if index > 9 else '00') + str(i - 9))
        column_welm = 'welm_' + ('' if index > 99 else ('0' if index > 9 else '00') + str(i - 9))

        df_train[column_elm] = train_y_hat_elm
        df_train[column_welm] = train_y_hat_welm

        df_test[column_elm] = test_y_hat_elm
        df_test[column_welm] = test_y_hat_welm

        df_test_t[column_elm] = y_test_hat_elm
        df_test_t[column_welm] = y_test_hat_welm

del df_train['id']
#df_train = pd.concat([df_train, pd.DataFrame(train_x, columns=columns[1:])], axis=1).as_matrix()
df_train = df_train.as_matrix()

del df_test['id']
#df_test = pd.concat([df_test, pd.DataFrame(test_x, columns=columns[1:])], axis=1).as_matrix()
df_test = df_test.as_matrix()

del df_test_t['id']
#df_test_t = pd.concat([df_test_t, pd.DataFrame(x_test, columns=columns[1:])], axis=1).as_matrix()
df_test_t = df_test_t.as_matrix()

print("Training...")

params = {'eta': 0.02,
          'max_depth': 4,
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'seed': 99,
          'silent': True}

d_train = xgb.DMatrix(df_train, train_y[:,0])

watchlist = [(xgb.DMatrix(df_train, train_y[:,0]), 'train'), (xgb.DMatrix(df_test_t, y_test), 'valid')]
xgm = xgb.train(params, d_train, 5000, watchlist, verbose_eval=1, early_stopping_rounds=100)


#
# Testing model
#
print("Predicting...")

test_y_hat = xgm.predict(xgb.DMatrix(test_x), ntree_limit=xgm.best_ntree_limit)

pd.DataFrame({'id': test_id, 'target': test_y_hat}).to_csv('./output/prediction/' + strftime("%Y%m%d%H%M%S", gmtime()) + '-xgb-porto-seguro.csv', index=False)
