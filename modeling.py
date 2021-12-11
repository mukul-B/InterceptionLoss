
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


columns_list = ["startDateTime", "EF", "GH", "SH", "DF", "MF", "PH", "WW", "Lai_500m", "MCH", "Biomass", "duration", "p", "IL"] #"startDateTime",
data = pd.read_csv('resource/Staging/output2.csv', names=columns_list)


X=data[["EF", "GH", "SH", "DF", "MF", "PH", "WW"]]
y=data[["IL"]]


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1)


regr = MLPRegressor(random_state=1, max_iter=1000,hidden_layer_sizes=(20,100)).fit(X_train, y_train)
y_pred = regr.predict(X_test)
ILdf = pd.DataFrame(y_pred)


print('regression score training', regr.score(X_train, y_train))
print('regression score test', regr.score(X_test, y_test))

#create correlation heatmap
data2 = data
data2.columns= ['startDateTime', 'LAI', "EF", "GH", "SH", "DF", "MF", "PH", "WW", 'MCH', 'Biomass', 'Duration', 'Precip', 'IL']
del data2['startDateTime']
print(data2.head())
corr = data2.corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
#cmap = sb.diverging_palette(20, 230, as_cmap=True)
sb.color_palette("crest_r", as_cmap=True)

sb.heatmap(corr, cmap="crest", vmax=1, vmin=-.2, annot=True)
sb.pairplot(data, hue='IL', height=2, palette="crest_r", diag_kind='hist')
print(data2.describe())

errors = mean_absolute_error(y_test, y_pred)
error2 = mean_squared_error(y_test, y_pred)
#error2 = math.sqrt(errors)
print('errors', errors, error2)

plt.show()
