from __future__ import print_function
from minepy import MINE

import pandas as pd

def mic(data):

   mine = MINE()

   losses = data[data['target'] == 1]
   non_losses = data[data['target'] == 0].sample(len(losses))

   del data

   all = pd.concat([losses, non_losses])

   del losses, non_losses

   y = all['target'].as_matrix()

   del all['target']

   print('var,mic')
   for _, column in all.iteritems():
      x = column.as_matrix()
      mine.compute_score(x, y)
      print(column.name, ',', mine.mic())
