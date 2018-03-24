import pandas as pd
import numpy as np

def recon(reg):
    integer = int(np.round((40*reg)**2)) # gives 2060 for our example
    for f in range(28):
        if (integer - f) % 27 == 0:
            F = f
    M = (integer - F)//27
    return F, M

def load_train_data_with_sample(columns=None):

    sample = pd.read_csv('./input/ds_sampling02_kaggle_porto_seguro_train.csv', sep=';')['id']

    train = pd.read_csv('./input/train.csv', sep=',')

    train = train.loc[train['id'].isin(sample.as_matrix())]

    train['ft_x1'] = train['ps_reg_01'] * train['ps_reg_03'] * train['ps_reg_02']
    train['ft_x2'] = train['ps_car_13'] * train['ps_reg_03'] * train['ps_car_13']
    train['ft_x3'] = train['ps_ind_03'] * train['ps_ind_15']

    features_dis = pd.read_csv('./output/ft_distance_vars_train.csv', sep=',')
    features_elm = pd.read_csv('./output/ft_elm_train.csv', sep=',')
    #features_mlp = pd.read_csv('./output/ft_mlp_train.csv', sep=',')
    # features_dae = pd.read_csv('./output/ft_dae_vars2_train.csv', sep=',')
    #features_psi = pd.read_csv('./output/ft_ps_id_train.csv', sep=',')
    features_kin = pd.read_csv('./output/ft_kinetic_train.csv', sep=',')
    features_pca = pd.read_csv('./output/ft_pca_train.csv', sep=',')
    features_rad = pd.read_csv('./output/ft_xy_train.csv', sep=',')

    train = train.merge(features_dis, on='id', how='inner')
    #train = train.merge(features_elm, on='id', how='inner')
    train = train.merge(features_kin, on='id', how='inner')
    #train = train.merge(features_mlp, on='id', how='inner')
    #train = train.merge(features_dae, on='id', how='inner')
    #train = train.merge(features_psi, on='id', how='inner')
    train = train.merge(features_pca, on='id', how='inner')
    train = train.merge(features_rad, on='id', how='inner')

    train_y = np.array([train['target'].as_matrix()]).T

    train_ids = train['id'].as_matrix()

    if columns is not None:
        train = train[columns]

    del train['target']

    train_x = train.as_matrix()

    return train_x, train_y, train_ids

def load_train_data(columns=None):

    train = pd.read_csv('./input/train.csv', sep=',')

    train['ft_x1'] = train['ps_reg_01'] * train['ps_reg_03'] * train['ps_reg_02']
    train['ft_x2'] = train['ps_car_13'] * train['ps_reg_03'] * train['ps_car_13']
    train['ft_x3'] = train['ps_ind_03'] * train['ps_ind_15']

    features_dis = pd.read_csv('./output/ft_distance_vars_train.csv', sep=',')
    features_elm = pd.read_csv('./output/ft_elm_train.csv', sep=',')
    #features_mlp = pd.read_csv('./output/ft_mlp_train.csv', sep=',')
    #features_xgb = pd.read_csv('./output/ft_xgb_train.csv', sep=',')
    #features_dae = pd.read_csv('./output/ft_dae_vars2_train.csv', sep=',')
    #features_psi = pd.read_csv('./output/ft_ps_id_train.csv', sep=',')
    features_rad = pd.read_csv('./output/ft_xy_train.csv', sep=',')
    features_pca = pd.read_csv('./output/ft_pca_vars_train.csv', sep=',')

    train = train.merge(features_dis, on='id', how='inner')
    train = train.merge(features_elm, on='id', how='inner')
    train = train.merge(features_rad, on='id', how='inner')
    train = train.merge(features_pca, on='id', how='inner')
    # train = train.merge(features_mlp, on='id', how='inner')
    #train = train.merge(features_dae, on='id', how='inner')
    #train = train.merge(features_xgb, on='id', how='inner')
    #train = train.merge(features_dae, on='id', how='inner')

    train_y = np.array([train['target'].as_matrix()]).T

    train_ids = train['id'].as_matrix()

    if columns is not None:
        train = train[columns]

    del train['target']

    train_x = train.as_matrix()

    return train_x, train_y, train_ids


def load_test_data(columns=None):

    test = pd.read_csv('./input/test.csv', sep=',')

    test['ft_x1'] = test['ps_reg_01'] * test['ps_reg_03'] * test['ps_reg_02']
    test['ft_x2'] = test['ps_car_13'] * test['ps_reg_03'] * test['ps_car_13']
    test['ft_x3'] = test['ps_ind_03'] * test['ps_ind_15']

    features_dis = pd.read_csv('./output/ft_distance_vars_test.csv', sep=',')
    features_pca = pd.read_csv('./output/ft_pca_vars_test.csv', sep=',')
    features_kin = pd.read_csv('./output/ft_kinetic_test.csv', sep=',')
    # features_elm = pd.read_csv('./output/ft_elm_test.csv', sep=',')
    # features_mlp = pd.read_csv('./output/ft_mlp_test.csv', sep=',')
    # features_xgb = pd.read_csv('./output/ft_xgb_test.csv', sep=',')
    # features_dae = pd.read_csv('./output/ft_dae_vars2_test.csv', sep=',')
    # features_psi = pd.read_csv('./output/ft_ps_id_test.csv', sep=',')
    # features_pca = pd.read_csv('./output/ft_pca_test.csv', sep=',')
    # features_rad = pd.read_csv('./output/ft_xy_test.csv', sep=',')

    test = test.merge(features_dis, on='id', how='inner')
    test = test.merge(features_pca, on='id', how='inner')

    test = test.merge(features_kin, on='id', how='inner')
    #test = test.merge(features_mlp, on='id', how='inner')
    #test = test.merge(features_xgb, on='id', how='inner')
    #test = test.merge(features_dae, on='id', how='inner')
    #test = test.merge(features_psi, on='id', how='inner')


    test_id = test['id'].as_matrix()

    if columns is not None:
        test = test[columns]

    test_x = test.as_matrix()

    return test_x, test_id

def load_train_test_sample(columns=None, training_fraction=.8):

    data = pd.read_csv('./input/train.csv', sep=',')

    data['ps_reg_F'] = data['ps_reg_03'].apply(lambda x: recon(x)[0])
    data['ps_reg_M'] = data['ps_reg_03'].apply(lambda x: recon(x)[1])

    data['ft_x1'] = data['ps_reg_01'] * data['ps_reg_03'] * data['ps_reg_02']
    data['ft_x2'] = data['ps_car_13'] * data['ps_reg_03'] * data['ps_car_13']
    data['ft_x3'] = data['ps_ind_03'] * data['ps_ind_15']

    features_dis = pd.read_csv('./output/ft_distance_vars_train.csv', sep=',')
    features_elm = pd.read_csv('./output/ft_elm_train.csv', sep=',')
    features_kin = pd.read_csv('./output/ft_kinetic_train.csv', sep=',')
    #features_mlp = pd.read_csv('./output/ft_mlp_train.csv', sep=',')
    #features_xgb = pd.read_csv('./output/ft_xgb_train.csv', sep=',')
    #features_dae = pd.read_csv('./output/ft_dae_vars2_train.csv', sep=',')
    #features_psi = pd.read_csv('./output/ft_ps_id_train.csv', sep=',')
    features_pca = pd.read_csv('./output/ft_pca_vars_train.csv', sep=',')
    features_rad = pd.read_csv('./output/ft_xy_train.csv', sep=',')

    x = data.loc[data['target'] == 0]
    y = data.loc[data['target'] == 1]

    train = x.sample(int(len(x.index) * training_fraction), random_state=49)
    train = train.append(y.sample(int(len(y.index) * training_fraction), random_state=49))

    train = train.merge(features_dis, on='id', how='inner')
   #train = train.merge(features_elm, on='id', how='inner')
    #train = train.merge(features_mlp, on='id', how='inner')
    #train = train.merge(features_xgb, on='id', how='inner')
    #train = train.merge(features_dae, on='id', how='inner')
    #train = train.merge(features_psi, on='id', how='inner')
    train = train.merge(features_pca, on='id', how='inner')
    train = train.merge(features_rad, on='id', how='inner')
    train = train.merge(features_kin, on='id', how='inner')

    train_y = np.array([train['target'].as_matrix()]).T

    train_id = train['id'].as_matrix()

    if columns is not None:
        train = train[columns]

    del train['target']

    train_x = train.as_matrix()

    test = data.loc[~data['id'].isin(train_id)]
    test = test.merge(features_dis, on='id', how='inner')
    test = test.merge(features_elm, on='id', how='inner')
    #test = test.merge(features_mlp, on='id', how='inner')
    #test = test.merge(features_xgb, on='id', how='inner')
    #test = test.merge(features_dae, on='id', how='inner')
    #test = test.merge(features_psi, on='id', how='inner')
    test = test.merge(features_pca, on='id', how='inner')
    test = test.merge(features_rad, on='id', how='inner')
    test = test.merge(features_kin, on='id', how='inner')

    test_id = test['id']

    test_y = np.array([test['target'].as_matrix()]).T

    if columns is not None:
        test = test[columns]

    del test['target']

    test_x = test.as_matrix()

    return train_x, train_y, test_x, test_y, train_id, test_id
