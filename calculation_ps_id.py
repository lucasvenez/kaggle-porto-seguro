import pandas as pd
import features as fet
from minepy import MINE

psid = fet.PSID()
mine = MINE()

data = pd.read_csv('input/train.csv', sep=',')

train_ps_id = psid.generate_id(data)

train_ps_id_df = pd.DataFrame({'ft_ps_id': train_ps_id})

train_ps_id_df['id'] = data['id']

train_ps_id_df.to_csv('output/ft_ps_id_train.csv')
