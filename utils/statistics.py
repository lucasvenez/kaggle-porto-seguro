from __future__ import print_function

def summary(dataframe):
   print('Number of Rows: ', len(dataframe))
   print('Number of Columns: ', len(dataframe.columns))
   print('Number of Losses: ', dataframe['target'].sum())

   print('Losses Percentage: ', float(dataframe['target'].sum()) / len(dataframe), '%')
   print('Regular Percentage: ', (len(dataframe) - float(dataframe['target'].sum())) / len(dataframe), '%')
