import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#for decision tree
from sklearn.ensemble import RandomForestRegressor


#importing datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#training random forest
regressor = RandomForestRegressor(n_estimators= 10, random_state=0)
regressor.fit(X,y)

#predicting the result
print(regressor.predict([[6.5]]))

#plotting the graph
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("truth or bluff : tree")
plt.xlabel("position level")
plt.ylabel('salary')
plt.show()