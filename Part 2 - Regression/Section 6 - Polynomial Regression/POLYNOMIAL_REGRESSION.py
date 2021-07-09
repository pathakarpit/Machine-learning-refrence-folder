import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#for splitting datasets
from sklearn.model_selection import train_test_split

#to train linear regression models
from sklearn.linear_model import LinearRegression

#for polynomial regression
from sklearn.preprocessing import PolynomialFeatures

#importing datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#splitting into test set and training set
#X_train, X_test, y_test, y_train = train_test_split(X, y, test_size = 0.2, random_state=0)

#training linear regresion model
lin_reg = LinearRegression()
lin_reg.fit(X, y)


#making polynomial regression
poly_reg = PolynomialFeatures(degree = 7)#its not necessary that with higher degree we will always get better results
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#printing the results of both functions
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("truth or bluff : linear")
plt.xlabel("position level")
plt.ylabel('salary')
plt.show()

plt.scatter(X,y,color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title("truth or bluff : poly")
plt.xlabel("position level")
plt.ylabel('salary')
plt.show()

#for better resolutions and smoother curve 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("truth or bluff : poly smoother")
plt.xlabel("position level")
plt.ylabel('salary')
plt.show()

#predicting new result with linear
print(lin_reg.predict([[6.5]]))

#predicting new result with polynomial
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))