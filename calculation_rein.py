from minepy import MINE
import pandas as pd

print('Loading data...')
data = pd.read_csv('./input/train.csv')[['id', 'target']]

ps_id = pd.read_csv('./output/ft_ps_id_train.csv')[['id', 'ft_ps_id']]

ps_id = ps_id.merge(data, on='id', how='inner')

value_max = 0

ps_id_unique =  pd.DataFrame({'ft_ps_id': ps_id['ft_ps_id'].unique()})

print('Calculating feature...')
for index, row in ps_id_unique.iterrows():

    sum   = ps_id.loc[ps_id['ft_ps_id'] == row['ft_ps_id']]['target'].sum()
    count = ps_id.loc[ps_id['ft_ps_id'] == row['ft_ps_id']].count()[1]
    value = 0. if count == 0 else sum / count

    ps_id_unique.loc[index, 'ft_ps_id_rei'] = value
    value_max = max(value_max, value)

    if index % 100 == 0:
        print('Calculating {} of {}. Max value: {}'.format(index + 1, ps_id_unique.shape[0], value_max))

print('Calculating MIC...')

df_fraud = ps_id.merge(ps_id_unique, on='ft_ps_id', how='inner').loc[ps_id['target'] == 1]
df_regul = ps_id.merge(ps_id_unique, on='ft_ps_id', how='inner').loc[ps_id['target'] == 0].sample(df_fraud.shape[0])

df = df_fraud.append(df_regul)

mine = MINE()

mine.compute_score(df['ft_ps_id'].as_matrix(), df['target'].as_matrix())
print(mine.mic())