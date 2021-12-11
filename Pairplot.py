import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pip install pinguoin
from pingouin import ancova

columns_list = ["startDateTime", "Lai_500m", "MCH", "Biomass", "duration", "p", "IL"] #"startDateTime",
data = pd.read_csv('resource/Staging/output2.csv', names=columns_list)

data2 = data
data2.columns= ['startDateTime','LAI', 'MCH', 'Biomass', 'Duration', 'Precip', 'IL']
del data2['startDateTime']
print(data2.head())

#sns.pairplot(data2, hue='IL', height=2, palette="crest_r", diag_kind='hist')

ancova(data = data2, dv = 'IL', covar = 'Precip')

#plt.show()