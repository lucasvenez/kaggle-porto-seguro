import data
import model
import pandas as pd

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
           'ft_pca'
           ]

#
# Loading data...
#
print("Loading data...")
train_x, train_y, train_id = data.load_train_data(columns)

print('Training...')
dae = model.DualAutoencoder(len(columns)-1, [40, 40, 40, 40])
dae.optimize(train_x, steps=10000, batch_size=10000)

feature_columns = ['ft_dae_' + ('0' + str(i) if i < 10 else str(i)) for i in range(1, 81)]

print('Predicting train...')
predict = pd.DataFrame(dae.predict(train_x), columns=feature_columns)
predict['id'] = pd.DataFrame(train_id, columns=['id'])

print('Exporting train...')
predict.to_csv('./output/ft_dae_vars2_train.csv', sep=',', index=False)

print('Predicting test...')
test_x, test_id = data.load_test_data(columns[1:])

predict = pd.DataFrame(dae.predict(test_x), columns=feature_columns)
predict['id'] = pd.DataFrame(test_id, columns=['id'])

print('Exporting test...')
predict.to_csv('./output/ft_dae_vars2_test.csv', sep=',', index=False)