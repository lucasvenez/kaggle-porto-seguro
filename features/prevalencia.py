import data
import pandas as pd
from minepy import MINE

train = data.load_train_data()

categorical_columns = [col for col in train if col.endswith('_cat')]

mine = MINE()

for column in categorical_columns:

    group = train.groupby([train[column]])

    result = pd.DataFrame(group.sum()['target'] / group.count()['target'])

    result = pd.DataFrame({column: result['target'].index, column + '_prevalence': result.reset_index()['target']})

    result = train.merge(result, how='inner', on=column)[[column, 'target']]

    fraud = result[result['target'] == 0].sample(len(result[result['target'] == 1]))
    regul = result[result['target'] == 1]

    result = pd.concat([fraud, regul]).as_matrix()

    mine.compute_score(result[:,0], result[:,1])

    print(column + ": " + str(mine.mic()))
