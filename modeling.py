
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


columns_list = ["startDateTime","Lai_500m", "MCH", "Biomass", "duration", "p", "IL"]
data = pd.read_csv('resource/Staging/output2.csv',names=columns_list)

data =data[:810]

X=data[["Lai_500m", "MCH", "Biomass", "duration", "p"]]
y=data[["IL"]]

print(data.head())
#
# X, y = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1)

print(len(X_test),len(X_train),len(X))
regr = MLPRegressor(random_state=1, max_iter=500,hidden_layer_sizes=(1,1)).fit(X_train, y_train)
print(X_test[:2],y_test[:2],regr.predict(X_test[:2]))

print(regr.score(X_test, y_test))

# plt.close()
#
# sns.set_style("whitegrid")
#
# sns.pairplot(data, hue="IL", height=2)
#
# # data.plot(x="IL", y=["p"])
# plt.show()