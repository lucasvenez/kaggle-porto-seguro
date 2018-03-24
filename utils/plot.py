from pandas.tools.plotting import radviz
from matplotlib import pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

def rad(dataframe):
   plt.figure()
   radviz(dataframe, 'target')
   plt.show()
