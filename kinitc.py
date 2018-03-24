import numpy as np
import pandas as pd
import xgboost as xgb
from multiprocessing import *
from sklearn import model_selection

#
print('Loading data...')
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

distance_train = pd.read_csv('./output/ft_distance_train.csv')
distance_test = pd.read_csv('./output/ft_distance_test.csv')

train = train.merge(distance_train, on='id', how='inner')
test = test.merge(distance_test, on='id', how='inner')

# more about kinetic features  developed  by Daia Alexandru    here  on the next  blog  please  read  last article :
# https://alexandrudaia.quora.com/

print('Calculating feature...')
##############################################creatinng   kinetic features for  train #####################################################
def kinetic(row):
    probs = np.unique(row, return_counts=True)[1] / len(row)
    kinetic = np.sum(probs ** 2)
    return kinetic

first_kin_names = [col for col in train.columns if '_ind_' in col]

subset_ind = train[first_kin_names]

kinetic_1 = []

for row in range(subset_ind.shape[0]):

    row = subset_ind.iloc[row]

    k = kinetic(row)

    kinetic_1.append(k)

second_kin_names = [col for col in train.columns if '_car_' in col and col.endswith('cat')]

subset_ind = train[second_kin_names]

kinetic_2 = []

for row in range(subset_ind.shape[0]):

    row = subset_ind.iloc[row]

    k = kinetic(row)

    kinetic_2.append(k)

third_kin_names = [col for col in train.columns if '_calc_' in col and not col.endswith('bin')]

subset_ind = train[second_kin_names]

kinetic_3 = []

for row in range(subset_ind.shape[0]):

    row = subset_ind.iloc[row]

    k = kinetic(row)

    kinetic_3.append(k)

fd_kin_names = [col for col in train.columns if '_calc_' in col and col.endswith('bin')]

subset_ind = train[fd_kin_names]

kinetic_4 = []

for row in range(subset_ind.shape[0]):

    row = subset_ind.iloc[row]

    k = kinetic(row)

    kinetic_4.append(k)

train['kinetic_1'] = np.array(kinetic_1)

train['kinetic_2'] = np.array(kinetic_2)

train['kinetic_3'] = np.array(kinetic_3)

train['kinetic_4'] = np.array(kinetic_4)

print('Exporting train kinect...')

train[['id', 'kinetic_1', 'kinetic_2', 'kinetic_3', 'kinetic_4']].to_csv('./output/ft_kinetic_train.csv', index=False, sep=',')

############################################reatinng   kinetic features for  test###############################################################


first_kin_names = [col for col in test.columns if '_ind_' in col]

subset_ind = test[first_kin_names]

kinetic_1 = []

for row in range(subset_ind.shape[0]):

    row = subset_ind.iloc[row]

    k = kinetic(row)

    kinetic_1.append(k)

second_kin_names = [col for col in test.columns if '_car_' in col and col.endswith('cat')]

subset_ind = test[second_kin_names]

kinetic_2 = []

for row in range(subset_ind.shape[0]):

    row = subset_ind.iloc[row]

    k = kinetic(row)

    kinetic_2.append(k)

third_kin_names = [col for col in test.columns if '_calc_' in col and not col.endswith('bin')]

subset_ind = test[second_kin_names]

kinetic_3 = []

for row in range(subset_ind.shape[0]):

    row = subset_ind.iloc[row]

    k = kinetic(row)

    kinetic_3.append(k)

fd_kin_names = [col for col in test.columns if '_calc_' in col and col.endswith('bin')]

subset_ind = test[fd_kin_names]

kinetic_4 = []

for row in range(subset_ind.shape[0]):

    row = subset_ind.iloc[row]

    k = kinetic(row)

    kinetic_4.append(k)

test['kinetic_1'] = np.array(kinetic_1)

test['kinetic_2'] = np.array(kinetic_2)

test['kinetic_3'] = np.array(kinetic_3)

test['kinetic_4'] = np.array(kinetic_4)

test[['id', 'kinetic_1', 'kinetic_2', 'kinetic_3', 'kinetic_4']].to_csv('./output/ft_kinetic_test.csv', index=False, sep=',')

##################################################################end  of kinetics ############################################################################

d_median = train.median(axis=0)

d_mean = train.mean(axis=0)

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id', 'target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol] == -1).values, axis=1)
    for c in dcol:
        if '_bin' not in c:
            df[c + str('_median_range')] = (df[c].values > d_median[c]).astype(int)
            df[c + str('_mean_range')] = (df[c].values > d_mean[c]).astype(int)
    return df


def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close();
    p.join()
    print('After Shape: ', df.shape)
    return df


def gini(y, pred):

    g = np.asarray(np.c_[y, pred, np.arange(len(y))], dtype=np.float)

    g = g[np.lexsort((g[:, 2], -1 * g[:, 1]))]

    gs = g[:, 0].cumsum().sum() / g[:, 0].sum()

    gs -= (len(y) + 1) / 2.

    return gs / len(y)

print('Selecting model...')

def gini_xgb(pred, y):
    return 'gini', gini(y, pred) / gini(y, y)

params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], test_size=0.25, random_state=99)

print('Transforming vars...')

x1 = multi_transform(x1)

y1 = x1['target']

x2 = multi_transform(x2)

y2 = x2['target']

test = multi_transform(test)

col = [c for c in x1.columns if c not in ['id','target']]

x1 = x1[col]

x2 = x2[col]

print('Training...')

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, verbose_eval=50, early_stopping_rounds=100)

print('Predicting and exporting...')

print(gini_xgb(model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), y2))

test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)

test[['id','target']].to_csv('uberKinetics.csv', index=False, float_format='%.5f')